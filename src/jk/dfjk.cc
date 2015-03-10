#include <math.h>
#include <string.h>
#include <core/basisset.h>
#include <mints/schwarz.h>
#include <df/df.h>
#include "jk.h"
#include <boost/timer/timer.hpp>

#if defined(_OPENMP)
#include <omp.h>
#endif

using namespace ambit;

namespace lightspeed {

DFJK::DFJK(
    const std::shared_ptr<SchwarzSieve>& sieve,
    const std::shared_ptr<SBasisSet>& auxiliary) :
    JK(sieve),
    auxiliary_(auxiliary)
{
}
void DFJK::initialize()
{
    AODFERI df(sieve_,auxiliary_);
    df.set_doubles(doubles());
    df.set_a(a());
    df.set_b(b());
    df.set_w(w());
    df.set_metric_condition(metric_condition());

    if (!force_disk() && doubles() >= df.ao_task_core_doubles()) {
        is_core_ = true;
        df_tensor_ = df.compute_ao_task_core();
    } else {
        is_core_ = false;
        df_tensor_ = df.compute_ao_task_disk();
    }
    initialized_ = true;
}
void DFJK::print(FILE* fh) const
{
    fprintf(fh, "  DFJK:\n");
    fprintf(fh, "    Primary Basis    = %14s\n", primary_->name().c_str());
    fprintf(fh, "    Auxiliary Basis  = %14s\n", auxiliary_->name().c_str());
    fprintf(fh, "    Doubles          = %14zu\n", doubles_);
    fprintf(fh, "    Force Disk?      = %14s\n", force_disk_ ? "Yes" : "No");
    fprintf(fh, "    Algorithm        = %14s\n", is_core_ ? "Core" : "Disk");
    fprintf(fh, "    Compute J?       = %14s\n", compute_J_ ? "Yes" : "No");
    fprintf(fh, "    Compute K?       = %14s\n", compute_K_ ? "Yes" : "No");
    fprintf(fh, "    Operator a       = %14.6E\n", a_);
    fprintf(fh, "    Operator b       = %14.6E\n", b_);
    fprintf(fh, "    Operator w       = %14.6E\n", w_);
    fprintf(fh, "    Product Cutoff   = %14.3E\n", product_cutoff_);
    fprintf(fh, "    Integral Cutoff  = %14.3E\n", sieve_->cutoff());
    fprintf(fh, "    Metric Condition = %14.3E\n", metric_condition_);
    fprintf(fh, "\n");
}
void DFJK::compute_JK_from_C(
    const std::vector<Tensor>& Ls,
    const std::vector<Tensor>& Rs,
    std::vector<Tensor>& Js,
    std::vector<Tensor>& Ks,
    const std::vector<double>& scaleJ,
    const std::vector<double>& scaleK)
{
    // => Check Initialization <= //

    if (!initialized_) throw std::runtime_error("DFJK: Call initalize before computing J/K matrices");

    // => Scale Defaults <= //

    std::vector<double> sJ = (scaleJ.size() ? scaleJ : std::vector<double>(Ls.size(),1.0));
    std::vector<double> sK = (scaleK.size() ? scaleK : std::vector<double>(Ls.size(),1.0));

    // => Tasking <= //

    bool JK_symm = true;
    for (size_t ind = 0; ind < Ls.size(); ind++) {
        JK_symm = JK_symm && Ls[ind] == Rs[ind];
    }
    bool do_J = compute_J_;
    bool do_K = compute_K_;
    if (!do_J && !do_K) return;

    // => Sizing <= //

    size_t naux    = auxiliary_->nfunction();
    size_t nbf     = primary_->nfunction();
    #if defined(_OPENMP)
    size_t nthread = omp_get_max_threads();
    #else
    size_t nthread = 1;
    #endif
    size_t max_nocc = 0;
    for (size_t ind = 0; ind < Ls.size(); ind++) {
        max_nocc = std::max(max_nocc, Ls[ind].dim(1));
    }

    // => Schwarz Layout <= //

    const std::vector<std::pair<int,int>>& shell_pairs = sieve_->shell_pairs();
    size_t npq = 0;
    for (size_t PQ = 0; PQ < shell_pairs.size(); PQ++) {
        int P = shell_pairs[PQ].first;
        int Q = shell_pairs[PQ].second;
        int nP = primary_->shell(P).nfunction();
        int nQ = primary_->shell(Q).nfunction();
        int oP = primary_->shell(P).function_index();
        int oQ = primary_->shell(Q).function_index();
        for (int p = 0; p < nP; p++) {
        for (int q = 0; q < nQ; q++) {
            npq++;
        }}
    }
    if (naux != df_tensor_.dim(0)) throw std::runtime_error("DFJK: DF tensor has incorrect size.");
    if (npq  != df_tensor_.dim(1)) throw std::runtime_error("DFJK: DF tensor has incorrect size.");

    // => Triangular D/J Matrices <= //

    std::vector<Tensor> Dtri;
    std::vector<Tensor> Jtri;
    if (do_J) {
        for (size_t ind = 0; ind < Ls.size(); ind++) {
            Tensor DF = Tensor::build(kCore, "DF", {nbf,nbf});
            DF("pq") = Ls[ind]("pi") * Rs[ind]("qi");
            Dtri.push_back(Tensor::build(kCore, "Dtri", {npq}));
            Jtri.push_back(Tensor::build(kCore, "Jtri", {npq}));
            double* DFp = DF.data().data();
            double* Dtp = Dtri[ind].data().data();
            size_t index = 0;
            for (size_t PQ = 0; PQ < shell_pairs.size(); PQ++) {
                int P = shell_pairs[PQ].first;
                int Q = shell_pairs[PQ].second;
                double perm = (P == Q ? 0.5 : 1.0);
                int nP = primary_->shell(P).nfunction();
                int nQ = primary_->shell(Q).nfunction();
                int oP = primary_->shell(P).function_index();
                int oQ = primary_->shell(Q).function_index();
                for (int p = 0; p < nP; p++) {
                for (int q = 0; q < nQ; q++) {
                    Dtp[index++] = perm * (
                        DFp[(p + oP)*nbf + (q + oQ)] +
                        DFp[(q + oQ)*nbf + (p + oP)]);
                }}
            }
        }
    }

    // => Memory Striping <= //

    size_t overhead = 0;

    overhead += (is_core_ ? naux * npq : 0);

    if (overhead > doubles()) throw std::runtime_error("DFJK: out of memory.");

    size_t rem = doubles() - overhead;

    size_t row_cost = 1;
    row_cost += (!is_core_ ? npq : 0); // B
    row_cost += (do_K ? nbf * nbf : 0); // C
    row_cost += (do_K ? (JK_symm ? 1L : 2) * nbf * max_nocc : 0); // E/F

    size_t max_naux = rem / row_cost;
    max_naux = std::min(naux,max_naux);

    if (max_naux < 1) throw std::runtime_error("DFJK: out of memory.");

    // => Effective 3-Index Tensor <= //

    Tensor B;
    size_t Boff = 0;
    if (is_core_) {
        B = df_tensor_;
    } else {
        B = Tensor::build(kCore,"B",{max_naux,npq});
    }

    // => J Buffers <= //

    Tensor d;
    if (do_J) {
        d = Tensor::build(kCore,"d",{max_naux});
    }

    // => K Buffers <= //

    Tensor C;
    Tensor E;
    Tensor F;
    if (do_K) {
        //printf("Buffers:\n");
        //boost::timer::auto_cpu_timer tval;
        C = Tensor::build(kCore,"C",{nbf,max_naux,nbf});
        E = Tensor::build(kCore,"E",{nbf,max_naux,max_nocc});
        if (JK_symm) {
            F = E;
        } else {
            F = Tensor::build(kCore,"F",{nbf,max_naux,max_nocc});
        }
    }

    // ==> Master Loop <== //

    for (size_t Astart = 0; Astart < naux; Astart += max_naux) {

        // => A Tasking <= //

        size_t Asize = (Astart + max_naux >= naux ? naux - Astart : max_naux);

        // => Disk Striping/Aliasing <= //

        if (is_core_) {
            Boff = Astart * npq;
        } else {
            B({{0,Asize},{0,npq}}) = df_tensor_({{Astart,Astart+Asize},{0,npq}});
            Boff = 0;
        }

        // => J <= //

        if (do_J) {
            //printf("J:\n");
            //boost::timer::auto_cpu_timer tval;
            for (size_t ind = 0; ind < Ls.size(); ind++) {
                d.gemm(B,Dtri[ind],false,false,Asize,1,npq,npq,1,1,Boff,0,0,1.0,0.0);
                Jtri[ind].gemm(B,d,true,false,npq,1,Asize,npq,1,1,Boff,0,0,sJ[ind],1.0);
            }
        }

        // => K <= //

        if (do_K) {

            //printf("K:\n");
            //boost::timer::auto_cpu_timer tval;
            
            // > Unpack B < //

            { // Timer
            //printf("K Unpack:\n");
            //boost::timer::auto_cpu_timer tval;

            double* Bp = B.data().data() + Boff;
            double* Cp = C.data().data();
            memset(Cp,'\0',sizeof(double)*nbf*Asize*nbf);
            //#pragma omp parallel for
            for (int A = 0; A < Asize; A++) {
                size_t pq = 0;
                for (size_t PQ = 0; PQ < shell_pairs.size(); PQ++) {
                    int P = shell_pairs[PQ].first;
                    int Q = shell_pairs[PQ].second;
                    int nP = primary_->shell(P).nfunction();
                    int nQ = primary_->shell(Q).nfunction();
                    int oP = primary_->shell(P).function_index();
                    int oQ = primary_->shell(Q).function_index();
                    for (int p = 0; p < nP; p++) {
                    for (int q = 0; q < nQ; q++) {
                        Cp[(p + oP) * Asize * nbf + A * nbf + (q + oQ)] =
                        Cp[(q + oQ) * Asize * nbf + A * nbf + (p + oP)] =
                        Bp[A*npq + pq];
                        pq++;
                    }}
                }
            }
            } // Timer
            
            // > Contractions < //

            { // Timer
            //printf("K Contract:\n");
            //boost::timer::auto_cpu_timer tval;
            for (size_t ind = 0; ind < Ls.size(); ind++) {
                size_t nocc = Ls[ind].dim(1);
                if (ind == 0 || Ls[ind] != Ls[ind-1]) {
                    E.gemm(C,Ls[ind],false,false,nbf*Asize,nocc,nbf,nbf,nocc,nocc,0,0,0,1.0,0.0);
                }
                if ((!JK_symm) && (ind == 0 || Rs[ind] != Rs[ind-1])) {
                    F.gemm(C,Rs[ind],false,false,nbf*Asize,nocc,nbf,nbf,nocc,nocc,0,0,0,1.0,0.0);
                }
                Ks[ind].gemm(E,F,false,true,nbf,nbf,Asize*nocc,Asize*nocc,Asize*nocc,nbf,0,0,0,sK[ind],1.0);
            }
            } // Timer
        }

    }

    // => J Extraction <= //

    if (do_J) {
        for (size_t ind = 0; ind < Ls.size(); ind++) {
            double* Jtp = Jtri[ind].data().data();
            double* Jp = Js[ind].data().data();
            size_t index = 0;
            for (size_t PQ = 0; PQ < shell_pairs.size(); PQ++) {
                int P = shell_pairs[PQ].first;
                int Q = shell_pairs[PQ].second;
                double perm = (P == Q ? 0.5 : 1.0);
                int nP = primary_->shell(P).nfunction();
                int nQ = primary_->shell(Q).nfunction();
                int oP = primary_->shell(P).function_index();
                int oQ = primary_->shell(Q).function_index();
                for (int p = 0; p < nP; p++) {
                for (int q = 0; q < nQ; q++) {
                    Jp[(p + oP) * nbf + (q + oQ)] += perm * Jtp[index];
                    Jp[(q + oQ) * nbf + (p + oP)] += perm * Jtp[index];
                    index++;
                }}
            }
        }
    }
}
void DFJK::compute_JK_from_D(
    const std::vector<Tensor>& Ds,
    const std::vector<bool>& symm,
    std::vector<Tensor>& Js,
    std::vector<Tensor>& Ks,
    const std::vector<double>& scaleJ,
    const std::vector<double>& scaleK)
{
    throw std::runtime_error("DFJK: Cannot do this yet.");
}
void DFJK::finalize()
{
    df_tensor_ = Tensor();
    initialized_ = false;
}

} // namespace libgaussian
