#include <math.h>
#include <string.h>
#include <core/basisset.h>
#include <mints/schwarz.h>
#include <mints/int4c.h>
#include "jk.h"

#include <omp.h>

using namespace ambit;

namespace lightspeed {

DirectJK::DirectJK(const std::shared_ptr<SchwarzSieve>& sieve) :
    JK(sieve)
{
}
void DirectJK::print(FILE* fh) const 
{
    fprintf(fh, "  DirectJK:\n");
    fprintf(fh, "    Primary Basis   = %14s\n", primary_->name().c_str());
    fprintf(fh, "    Doubles         = %14zu\n", doubles_);
    fprintf(fh, "    Compute J?      = %14s\n", compute_J_ ? "Yes" : "No");
    fprintf(fh, "    Compute K?      = %14s\n", compute_K_ ? "Yes" : "No");
    fprintf(fh, "    Operator a      = %14.6E\n", a_);
    fprintf(fh, "    Operator b      = %14.6E\n", b_);
    fprintf(fh, "    Operator w      = %14.6E\n", w_);
    fprintf(fh, "    Product Cutoff  = %14.3E\n", product_cutoff_);
    fprintf(fh, "    Integral Cutoff = %14.3E\n", sieve_->cutoff());
    fprintf(fh, "\n");
}
void DirectJK::compute_JK_from_C(
    const std::vector<Tensor>& L,
    const std::vector<Tensor>& R,
    std::vector<Tensor>& J,
    std::vector<Tensor>& K,
    const std::vector<double>& scaleJ,
    const std::vector<double>& scaleK)
{
    std::vector<Tensor> D(L.size());
    std::vector<bool> symm(L.size());
    for (size_t ind = 0; ind < L.size(); ind++) {
        D[ind] = Tensor::build(kCore,"D",{L[ind].dim(0),R[ind].dim(0)});
        D[ind]("pq") = L[ind]("pi") * R[ind]("qi");
        symm[ind] = L[ind] == R[ind];
    }
    compute_JK_from_D(D,symm,J,K,scaleJ,scaleK);
}
void DirectJK::compute_JK_from_D(
    const std::vector<Tensor>& Ds,
    const std::vector<bool>& symm,
    std::vector<Tensor>& Js,
    std::vector<Tensor>& Ks,
    const std::vector<double>& scaleJ,
    const std::vector<double>& scaleK)
{ 
    // => Scale Defaults <= //

    std::vector<double> sJ = (scaleJ.size() ? scaleJ : std::vector<double>(Ds.size(),1.0));
    std::vector<double> sK = (scaleK.size() ? scaleK : std::vector<double>(Ds.size(),1.0));

    // => Tasking <= //
    
    bool JK_symm = true;
    for (bool v : symm) {
        JK_symm = JK_symm && v;
    }
    bool do_J = compute_J_;
    bool do_K = compute_K_;
    if (!do_J && !do_K) return; 

    // => Sizing <= //
    
    size_t natom   = primary_->natom();
    size_t nshell  = primary_->nshell();
    size_t nbf     = primary_->nfunction();
    size_t nthread = omp_get_max_threads();

    // => Atom Indices <= //

    const std::vector<std::vector<size_t>>& atoms_to_shells = primary_->atoms_to_shell_inds(); 
    size_t max_nbf = 0;
    for (int A = 0; A < natom; A++) {
        const std::vector<size_t>& Ainds = atoms_to_shells[A];
        size_t this_nbf = 0;
        for (int P = 0; P < Ainds.size(); P++) {
            this_nbf += primary_->shell(Ainds[P]).nfunction();
        }
        max_nbf = std::max(this_nbf, max_nbf);
    }
    size_t max_nbf2 = max_nbf * max_nbf;

    // => Significant Atom Pairs <= //

    std::vector<std::pair<int,int>> atom_pairs;
    for (int A = 0; A < natom; A++) {
        for (int B = 0; B <= A; B++) {
            const std::vector<size_t>& Ainds = atoms_to_shells[A];
            const std::vector<size_t>& Binds = atoms_to_shells[B];
            bool found = false;
            for (int P = 0; P < Ainds.size(); P++) {
                for (int Q = 0; Q < Binds.size(); Q++) {
                    if (sieve_->shell_estimate_PQPQ(Ainds[P], Binds[Q]) * sieve_->max_PQRS() >= sieve_->cutoff()) {
                        found = true;
                    }
                }
            }
            if (found) atom_pairs.push_back(std::pair<int,int>(A,B));
        }
    }
    size_t natom_task2 = atom_pairs.size();
    size_t natom_task4 = atom_pairs.size() * atom_pairs.size();

    // => Sources/Pointers <= //

    std::vector<const double*> Dsp(Ds.size());
    std::vector<double*> Jsp(Ds.size());
    std::vector<double*> Ksp(Ds.size());
    for (size_t ind = 0; ind < Ds.size(); ind++) {
        Dsp[ind] = Ds[ind].data().data();
        if (do_J) Jsp[ind] = Js[ind].data().data();
        if (do_K) Ksp[ind] = Ks[ind].data().data();
    }

    // => Maximum Shell Pair Values <= //

    Tensor DS = Tensor::build(kCore, "DS", {nshell, nshell});
    double* DSp = DS.data().data();
    for (size_t P = 0; P < nshell; P++) {
        for (size_t Q = 0; Q < nshell; Q++) {
            int nP = primary_->shell(P).nfunction();
            int nQ = primary_->shell(Q).nfunction();
            int oP = primary_->shell(P).function_index();
            int oQ = primary_->shell(Q).function_index();
            double val = 0.0;
            for (size_t ind = 0; ind < Ds.size(); ind++) {
                for (int p = 0; p < nP; p++) {
                    for (int q = 0; q < nQ; q++) {
                        val = std::max(val,Dsp[ind][(p + oP) * nbf + (q + oQ)]);
                    }
                }
            }
            DSp[P*nshell+Q] = val;
        }
    }

    // => Integrals <= //
    
    std::vector<std::shared_ptr<PotentialInt4C>> ints;
    for (int t = 0; t < nthread; t++) {
        ints.push_back(std::shared_ptr<PotentialInt4C>(new PotentialInt4C(
            primary_,primary_,primary_,primary_,0,a_,b_,w_)));
    } 

    // => Buffers <= //

    size_t ntemp = (JK_symm ? 6 : 10); // 2 J, 4 K, 4 K'
    std::vector<std::vector<double*> > JKT;
    for (int t = 0; t < nthread; t++) {
        JKT.push_back(std::vector<double*>(Ds.size()));
        for (int ind = 0; ind < Ds.size(); ind++) {
            JKT[t][ind] = new double[ntemp * max_nbf2];
        }
    }

    // ==> Master Loop <== //

    #pragma omp parallel for schedule(dynamic)
    for (size_t ABCDtask = 0; ABCDtask < natom_task4; ABCDtask++) {
        size_t AB = ABCDtask / natom_task2;
        size_t CD = ABCDtask % natom_task2;
        size_t A = atom_pairs[AB].first;
        size_t B = atom_pairs[AB].second;
        size_t C = atom_pairs[CD].first;
        size_t D = atom_pairs[CD].second;
    
        if (A < C) continue;

        const std::vector<size_t>& Pinds = atoms_to_shells[A];
        const std::vector<size_t>& Qinds = atoms_to_shells[B];
        const std::vector<size_t>& Rinds = atoms_to_shells[C];
        const std::vector<size_t>& Sinds = atoms_to_shells[D];

        size_t nPtask = Pinds.size();
        size_t nQtask = Qinds.size();
        size_t nRtask = Rinds.size();
        size_t nStask = Sinds.size();

        size_t P2start = primary_->shell(Pinds[0]).function_index();
        size_t Q2start = primary_->shell(Qinds[0]).function_index();
        size_t R2start = primary_->shell(Rinds[0]).function_index();
        size_t S2start = primary_->shell(Sinds[0]).function_index();

        size_t P2stop = (A + 1 == natom ? nbf : primary_->shell(atoms_to_shells[A+1][0]).function_index());
        size_t Q2stop = (B + 1 == natom ? nbf : primary_->shell(atoms_to_shells[B+1][0]).function_index());
        size_t R2stop = (C + 1 == natom ? nbf : primary_->shell(atoms_to_shells[C+1][0]).function_index());
        size_t S2stop = (D + 1 == natom ? nbf : primary_->shell(atoms_to_shells[D+1][0]).function_index());

        size_t dPsize = P2stop - P2start;
        size_t dQsize = Q2stop - Q2start;
        size_t dRsize = R2stop - R2start;
        size_t dSsize = S2stop - S2start;

        int t = omp_get_thread_num();

        // => Shell Quartets <= //

        bool touched = false;
        for (size_t P2 = 0; P2 < nPtask; P2++) {
        for (size_t Q2 = 0; Q2 < nQtask; Q2++) {
        for (size_t R2 = 0; R2 < nRtask; R2++) {
        for (size_t S2 = 0; S2 < nStask; S2++) {

            size_t P = Pinds[P2];
            size_t Q = Qinds[Q2];
            size_t R = Rinds[R2];
            size_t S = Sinds[S2];

            // => Permutational Screening <= //

            if (P < Q) continue;
            if (R < S) continue;
            if (P * nshell + Q < R * nshell + S) continue;

            // => Schwarz Screening <= // 

            double IPQRS = sieve_->shell_estimate_PQRS(P,Q,R,S); 
            if (IPQRS < sieve_->cutoff()) continue;

            // => Density Screening <= //

            bool product_significant = false;
            if (do_J) {
                if (DSp[P * nshell + Q] * IPQRS >= product_cutoff_) product_significant = true;
                if (DSp[Q * nshell + P] * IPQRS >= product_cutoff_) product_significant = true;
                if (DSp[R * nshell + S] * IPQRS >= product_cutoff_) product_significant = true;
                if (DSp[S * nshell + R] * IPQRS >= product_cutoff_) product_significant = true;
            } 
            if (do_K && !product_significant) {
                if (DSp[P * nshell + R] * IPQRS >= product_cutoff_) product_significant = true;
                if (DSp[P * nshell + S] * IPQRS >= product_cutoff_) product_significant = true;
                if (DSp[Q * nshell + R] * IPQRS >= product_cutoff_) product_significant = true;
                if (DSp[Q * nshell + S] * IPQRS >= product_cutoff_) product_significant = true;
            }
            if (do_K && !JK_symm && !product_significant) {
                if (DSp[R * nshell + P] * IPQRS >= product_cutoff_) product_significant = true;
                if (DSp[R * nshell + Q] * IPQRS >= product_cutoff_) product_significant = true;
                if (DSp[S * nshell + P] * IPQRS >= product_cutoff_) product_significant = true;
                if (DSp[S * nshell + S] * IPQRS >= product_cutoff_) product_significant = true;
            }
            if (!product_significant) continue;

            ints[t]->compute_shell(P,Q,R,S); 
            double* buffer = ints[t]->buffer();

            double prefactor = 
                (P == Q ? 0.5 : 1.0) * 
                (R == S ? 0.5 : 1.0) * 
                (P == R && Q == S ? 0.5 : 1.0);

            int Psize = primary_->shell(P).nfunction();
            int Qsize = primary_->shell(Q).nfunction();
            int Rsize = primary_->shell(R).nfunction();
            int Ssize = primary_->shell(S).nfunction();

            int Poff = primary_->shell(P).function_index();
            int Qoff = primary_->shell(Q).function_index();
            int Roff = primary_->shell(R).function_index();
            int Soff = primary_->shell(S).function_index();

            int Poff2 = Poff - P2start;
            int Qoff2 = Qoff - Q2start;
            int Soff2 = Soff - S2start;
            int Roff2 = Roff - R2start;

            for (size_t ind = 0; ind < Ds.size(); ind++) {
                const double* Dp = Dsp[ind];
                double* JKTp = JKT[t][ind];
                if (!touched) {
                    memset(&JKTp[0L * max_nbf2], '\0', dPsize * dQsize * sizeof(double));
                    memset(&JKTp[1L * max_nbf2], '\0', dRsize * dSsize * sizeof(double));
                    memset(&JKTp[2L * max_nbf2], '\0', dPsize * dRsize * sizeof(double));
                    memset(&JKTp[3L * max_nbf2], '\0', dPsize * dSsize * sizeof(double));
                    memset(&JKTp[4L * max_nbf2], '\0', dQsize * dRsize * sizeof(double));
                    memset(&JKTp[5L * max_nbf2], '\0', dQsize * dSsize * sizeof(double));
                    if (!JK_symm) {
                        memset(&JKTp[6L * max_nbf2], '\0', dRsize * dPsize * sizeof(double));
                        memset(&JKTp[7L * max_nbf2], '\0', dSsize * dPsize * sizeof(double));
                        memset(&JKTp[8L * max_nbf2], '\0', dRsize * dQsize * sizeof(double));
                        memset(&JKTp[9L * max_nbf2], '\0', dSsize * dQsize * sizeof(double));
                    }
                }
                double* buffer2 = buffer;
                double* J1p = &JKTp[0L * max_nbf2];
                double* J2p = &JKTp[1L * max_nbf2];
                double* K1p = &JKTp[2L * max_nbf2];
                double* K2p = &JKTp[3L * max_nbf2];
                double* K3p = &JKTp[4L * max_nbf2];
                double* K4p = &JKTp[5L * max_nbf2];
                double* K5p;
                double* K6p;
                double* K7p;
                double* K8p;
                if (!JK_symm) {
                    K5p = &JKTp[6L * max_nbf2];
                    K6p = &JKTp[7L * max_nbf2];
                    K7p = &JKTp[8L * max_nbf2];
                    K8p = &JKTp[9L * max_nbf2];
                }
                for (int p = 0; p < Psize; p++) {
                for (int q = 0; q < Qsize; q++) {
                for (int r = 0; r < Rsize; r++) {
                for (int s = 0; s < Ssize; s++) {
                    J1p[(p + Poff2) * dQsize + (q + Qoff2)] += prefactor * (Dp[(r + Roff) * nbf + (s + Soff)] + Dp[(s + Soff) * nbf + (r + Roff)]) * (*buffer2);
                    J2p[(r + Roff2) * dSsize + (s + Soff2)] += prefactor * (Dp[(p + Poff) * nbf + (q + Qoff)] + Dp[(q + Qoff) * nbf + (p + Poff)]) * (*buffer2);
                    K1p[(p + Poff2) * dRsize + (r + Roff2)] += prefactor * Dp[(q + Qoff) * nbf + (s + Soff)] * (*buffer2);
                    K2p[(p + Poff2) * dSsize + (s + Soff2)] += prefactor * Dp[(q + Qoff) * nbf + (r + Roff)] * (*buffer2);
                    K3p[(q + Qoff2) * dRsize + (r + Roff2)] += prefactor * Dp[(p + Poff) * nbf + (s + Soff)] * (*buffer2);
                    K4p[(q + Qoff2) * dSsize + (s + Soff2)] += prefactor * Dp[(p + Poff) * nbf + (r + Roff)] * (*buffer2);
                    if (!JK_symm) {
                        K5p[(r + Roff2) * dPsize + (p + Poff2)] += prefactor * Dp[(s + Soff) * nbf + (q + Qoff)] * (*buffer2);
                        K6p[(s + Soff2) * dPsize + (p + Poff2)] += prefactor * Dp[(r + Roff) * nbf + (q + Qoff)] * (*buffer2);
                        K7p[(r + Roff2) * dQsize + (q + Qoff2)] += prefactor * Dp[(s + Soff) * nbf + (p + Poff)] * (*buffer2);
                        K8p[(s + Soff2) * dQsize + (q + Qoff2)] += prefactor * Dp[(r + Roff) * nbf + (p + Poff)] * (*buffer2);
                    }
                    buffer2++;
                }}}}
            }
            touched = true;

        }}}}
        if (!touched) continue;

        // => Stripe out <= //

        for (int ind = 0; ind < Ds.size(); ind++) {
            double* JKTp = JKT[t][ind];
            double* Jp = Jsp[ind];
            double* Kp = Ksp[ind];

            double* J1p = &JKTp[0L * max_nbf2];
            double* J2p = &JKTp[1L * max_nbf2];
            double* K1p = &JKTp[2L * max_nbf2];
            double* K2p = &JKTp[3L * max_nbf2];
            double* K3p = &JKTp[4L * max_nbf2];
            double* K4p = &JKTp[5L * max_nbf2];
            double* K5p;
            double* K6p;
            double* K7p;
            double* K8p;
            if (!JK_symm) {
                K5p = &JKTp[6L * max_nbf2];
                K6p = &JKTp[7L * max_nbf2];
                K7p = &JKTp[8L * max_nbf2];
                K8p = &JKTp[9L * max_nbf2];
            }

            if (do_J) {

                // > J_PQ < //

                for (int P2 = 0; P2 < nPtask; P2++) {
                for (int Q2 = 0; Q2 < nQtask; Q2++) {
                    int P = Pinds[P2];
                    int Q = Qinds[Q2];
                    int Psize = primary_->shell(P).nfunction();
                    int Qsize = primary_->shell(Q).nfunction();
                    int Poff =  primary_->shell(P).function_index();
                    int Qoff =  primary_->shell(Q).function_index();
                    int Poff2 = Poff - P2start;
                    int Qoff2 = Qoff - Q2start;
                    for (int p = 0; p < Psize; p++) {
                    for (int q = 0; q < Qsize; q++) {
                        #pragma omp atomic
                        Jp[(p + Poff) * nbf + q + Qoff] += J1p[(p + Poff2) * dQsize + q + Qoff2];
                    }}
                }}

                // > J_RS < //

                for (int R2 = 0; R2 < nRtask; R2++) {
                for (int S2 = 0; S2 < nStask; S2++) {
                    int R = Rinds[R2];
                    int S = Sinds[S2];
                    int Rsize = primary_->shell(R).nfunction();
                    int Ssize = primary_->shell(S).nfunction();
                    int Roff =  primary_->shell(R).function_index();
                    int Soff =  primary_->shell(S).function_index();
                    int Roff2 = Roff - R2start;
                    int Soff2 = Soff - S2start;
                    for (int r = 0; r < Rsize; r++) {
                    for (int s = 0; s < Ssize; s++) {
                        #pragma omp atomic
                        Jp[(r + Roff) * nbf + s + Soff] += J2p[(r + Roff2) * dSsize + s + Soff2];
                    }}
                }}

            } 

            if (do_K) {

                // > K_PR < //

                for (int P2 = 0; P2 < nPtask; P2++) {
                for (int R2 = 0; R2 < nRtask; R2++) {
                    int P = Pinds[P2]; 
                    int R = Rinds[R2];
                    int Psize = primary_->shell(P).nfunction();
                    int Rsize = primary_->shell(R).nfunction();
                    int Poff =  primary_->shell(P).function_index();
                    int Roff =  primary_->shell(R).function_index();
                    int Poff2 = Poff - P2start;
                    int Roff2 = Roff - R2start;
                    for (int p = 0; p < Psize; p++) {
                    for (int r = 0; r < Rsize; r++) {
                        #pragma omp atomic
                        Kp[(p + Poff) * nbf + r + Roff] += K1p[(p + Poff2) * dRsize + r + Roff2];
                        if (!JK_symm) {
                            #pragma omp atomic
                            Kp[(r + Roff) * nbf + p + Poff] += K5p[(r + Roff2) * dPsize + p + Poff2];
                        }
                    }}
                }}

                // > K_PS < //

                for (int P2 = 0; P2 < nPtask; P2++) {
                for (int S2 = 0; S2 < nStask; S2++) {
                    int P = Pinds[P2];
                    int S = Sinds[S2];
                    int Psize = primary_->shell(P).nfunction();
                    int Ssize = primary_->shell(S).nfunction();
                    int Poff =  primary_->shell(P).function_index();
                    int Soff =  primary_->shell(S).function_index();
                    int Poff2 = Poff - P2start;
                    int Soff2 = Soff - S2start;
                    for (int p = 0; p < Psize; p++) {
                    for (int s = 0; s < Ssize; s++) {
                        #pragma omp atomic
                        Kp[(p + Poff) * nbf + s + Soff] += K2p[(p + Poff2) * dSsize + s + Soff2];
                        if (!JK_symm) {
                            #pragma omp atomic
                            Kp[(s + Soff) * nbf + p + Poff] += K6p[(s + Soff2) * dPsize + p + Poff2];
                        }
                    }}
                }}

                // > K_QR < //

                for (int Q2 = 0; Q2 < nQtask; Q2++) {
                for (int R2 = 0; R2 < nRtask; R2++) {
                    int Q = Qinds[Q2];
                    int R = Rinds[R2];
                    int Qsize = primary_->shell(Q).nfunction();
                    int Rsize = primary_->shell(R).nfunction();
                    int Qoff =  primary_->shell(Q).function_index();
                    int Roff =  primary_->shell(R).function_index();
                    int Qoff2 = Qoff - Q2start;
                    int Roff2 = Roff - R2start;
                    for (int q = 0; q < Qsize; q++) {
                    for (int r = 0; r < Rsize; r++) {
                        #pragma omp atomic
                        Kp[(q + Qoff) * nbf + r + Roff] += K3p[(q + Qoff2) * dRsize + r + Roff2];
                        if (!JK_symm) {
                            #pragma omp atomic
                            Kp[(r + Roff) * nbf + q + Qoff] += K7p[(r + Roff2) * dQsize + q + Qoff2];
                        }
                    }}
                }}

                // > K_QS < //

                for (int Q2 = 0; Q2 < nQtask; Q2++) {
                for (int S2 = 0; S2 < nStask; S2++) {
                    int Q = Qinds[Q2];
                    int S = Sinds[S2];
                    int Qsize = primary_->shell(Q).nfunction();
                    int Ssize = primary_->shell(S).nfunction();
                    int Qoff =  primary_->shell(Q).function_index();
                    int Soff =  primary_->shell(S).function_index();
                    int Qoff2 = Qoff - Q2start;
                    int Soff2 = Soff - S2start;
                    for (int q = 0; q < Qsize; q++) {
                    for (int s = 0; s < Ssize; s++) {
                        #pragma omp atomic
                        Kp[(q + Qoff) * nbf + s + Soff] += K4p[(q + Qoff2) * dSsize + s + Soff2];
                        if (!JK_symm) {
                            #pragma omp atomic
                            Kp[(s + Soff) * nbf + q + Qoff] += K8p[(s + Soff2) * dQsize + q + Qoff2];
                        }
                    }}
                }}

            }

        } // End stripe out
    }

    // => Free Buffers <= //

    for (int t = 0; t < nthread; t++) {
        for (int ind = 0; ind < Ds.size(); ind++) {
            delete[] JKT[t][ind];
        }
    }

    // => Hermitivitize <= //
    
    if (do_J) {
        for (size_t ind = 0; ind < Ds.size(); ind++) {
            double* J2p = Jsp[ind];
            for (int p = 0; p < nbf; p++) {
                for (int q = 0; q <= p; q++) {
                    J2p[p*nbf+q] = J2p[q*nbf+p] =
                    J2p[p*nbf+q] + J2p[q*nbf+p];
                }
            }
            Js[ind].scale(sJ[ind]);
        }
    } 

    if (do_K && JK_symm) {
        for (size_t ind = 0; ind < Ds.size(); ind++) {
            double* K2p = Ksp[ind];
            for (int p = 0; p < nbf; p++) {
                for (int q = 0; q <= p; q++) {
                    K2p[p*nbf+q] = K2p[q*nbf+p] =
                    K2p[p*nbf+q] + K2p[q*nbf+p];
                }
            }
            Ks[ind].scale(sK[ind]);
        }
    }
}

} // namespace libgaussian
