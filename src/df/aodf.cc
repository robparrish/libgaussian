#include <math.h>
#include <mints/int4c.h>
#include <mints/schwarz.h>
#include "df.h"

#include <omp.h>

using namespace ambit;

namespace lightspeed {

AODFERI::AODFERI(
    const std::shared_ptr<SchwarzSieve>& sieve,
    const std::shared_ptr<SBasisSet>& auxiliary) :
    DFERI(sieve,auxiliary)
{
}
size_t AODFERI::ao_task_core_doubles() const 
{
    size_t naux = auxiliary_->nfunction();
    const std::vector<std::pair<int,int>>& shell_pairs = sieve_->shell_pairs();
    size_t nPQshell = sieve_->shell_pairs().size();
    size_t npq = 0;
    std::vector<size_t> pqstarts(nPQshell);
    pqstarts[0] = 0;
    for (size_t PQ = 0; PQ < nPQshell; PQ++) {
        size_t P = shell_pairs[PQ].first;
        size_t Q = shell_pairs[PQ].second;
        size_t offset = primary_->shell(P).nfunction() * primary_->shell(Q).nfunction();
        if (PQ < nPQshell - 1) pqstarts[PQ + 1] = pqstarts[PQ] + offset;
        npq += offset;
    }
    
    return 2L * naux * naux + naux * npq;
}
size_t AODFERI::ao_task_disk_doubles() const 
{
    size_t naux = auxiliary_->nfunction();
    return 3L * naux; 
}
Tensor AODFERI::compute_ao_task_core(double power) const
{
    // => Pair state vector <= //

    const std::vector<std::pair<int,int>>& shell_pairs = sieve_->shell_pairs();

    // => Sizing <= //

    size_t nAshell = auxiliary_->nshell();
    size_t nPQshell = sieve_->shell_pairs().size();
    size_t naux = auxiliary_->nfunction();

    // => Pair indexing <= //

    size_t npq = 0;
    std::vector<size_t> pqstarts(nPQshell);
    pqstarts[0] = 0;
    for (size_t PQ = 0; PQ < nPQshell; PQ++) {
        size_t P = shell_pairs[PQ].first;
        size_t Q = shell_pairs[PQ].second;
        size_t offset = primary_->shell(P).nfunction() * primary_->shell(Q).nfunction();
        if (PQ < nPQshell - 1) pqstarts[PQ + 1] = pqstarts[PQ] + offset;
        npq += offset;
    }
    
    // => Memory Check <= //
    
    size_t required = ao_task_core_doubles();
    if (required > doubles()) throw std::runtime_error("AODFERI needs 2J + Apq memory for core.");

    // => Inverse Metric <= //
        
    Tensor J = metric_power_core(power,metric_condition_);

    // => Target <= //
    
    Tensor B = Tensor::build(kCore, "B", {naux, npq});
    double* Bp = B.data().data();

    // => Integrals <= //

    int nthread = omp_get_max_threads();    
    std::shared_ptr<SBasisSet> zero = SBasisSet::zero_basis();
    std::vector<std::shared_ptr<PotentialInt4C>> Bints;
    for (int t = 0; t < nthread; t++) {
        Bints.push_back(std::shared_ptr<PotentialInt4C>(new PotentialInt4C(auxiliary_,zero,primary_,primary_,0,a_,b_,w_)));
    }

    size_t nAPQtask = nAshell * nPQshell;
    #pragma omp parallel for schedule(dynamic)
    for (size_t APQtask = 0; APQtask < nAPQtask; APQtask++) {
        size_t A  = APQtask / nPQshell;
        size_t PQ = APQtask % nPQshell;
        size_t P = shell_pairs[PQ].first;
        size_t Q = shell_pairs[PQ].second;
        int nA = auxiliary_->shell(A).nfunction();
        int nP = primary_->shell(P).nfunction();
        int nQ = primary_->shell(Q).nfunction();
        size_t oA = auxiliary_->shell(A).function_index();
        size_t opq = pqstarts[PQ]; 
        int t = omp_get_thread_num();
        Bints[t]->compute_shell(A,0,P,Q);
        double* Bbuffer = Bints[t]->buffer();
        for (int a = 0; a < nA; a++) {
        for (int p = 0; p < nP; p++) {
        for (int q = 0; q < nQ; q++) {
            Bp[(a + oA) * npq + p*nQ + q + opq] = (*Bbuffer++);
        }}}
    }
    Bints.clear();
      
    // => Fitting <= //

    Tensor T = Tensor::build(kCore, "T", {naux, naux});     
    for (size_t pqstart = 0L; pqstart < npq; pqstart += naux) {
        size_t pqsize = (pqstart + naux >= npq ? npq - pqstart : naux);
        T({{0,naux},{0,pqsize}}) = B({{0,naux},{pqstart,pqstart+pqsize}});
        B.gemm(J,T,false,false,naux,pqsize,naux,naux,naux,npq,0,0,pqstart,1.0,0.0);
    }
    
    return B;
}
Tensor AODFERI::compute_ao_task_disk(double power) const
{
    // => Pair state vector <= //

    const std::vector<std::pair<int,int>>& shell_pairs = sieve_->shell_pairs();

    // => Sizing <= //

    size_t nAshell = auxiliary_->nshell();
    size_t nPQshell = sieve_->shell_pairs().size();
    size_t naux = auxiliary_->nfunction();

    // => Pair indexing <= //

    size_t npq = 0;
    std::vector<size_t> pqstarts(nPQshell);
    pqstarts[0] = 0;
    for (size_t PQ = 0; PQ < nPQshell; PQ++) {
        size_t P = shell_pairs[PQ].first;
        size_t Q = shell_pairs[PQ].second;
        size_t offset = primary_->shell(P).nfunction() * primary_->shell(Q).nfunction();
        if (PQ < nPQshell - 1) pqstarts[PQ + 1] = pqstarts[PQ] + offset;
        npq += offset;
    }
    
    // => Memory Check <= //
    
    size_t required = 0L;
    required += 2L * naux * naux;
    required += naux * primary_->max_nfunction() * primary_->max_nfunction(); 
    if (required > doubles()) throw std::runtime_error("AODFERI needs 2J + nA * maxp^2 memory for core.");

    size_t rem = doubles() - 2L * naux * naux;
    size_t maxpq = rem / naux;
    maxpq = std::min(npq, maxpq);

    // => Shell Pair Tasks <= //

    std::vector<size_t> PQtasks;
    PQtasks.push_back(0);
    for (size_t ind = 0, pqoff = 0; ind < shell_pairs.size(); ind++) {
        size_t pqstart = pqstarts[PQtasks.back()];
        size_t pqstop = pqstarts[ind];
        if (pqstop - pqstart > maxpq) {
            PQtasks.push_back(ind);
        }
    }
    PQtasks.push_back(shell_pairs.size());

    // => Inverse Metric <= //
        
    Tensor J = metric_power_core(power,metric_condition_);

    // => Target <= //
    
    Tensor B = Tensor::build(kDisk, "B", {naux, npq});
    
    // => Buffers <= //

    Tensor T = Tensor::build(kCore, "T", {naux, naux});     
    Tensor B2 = Tensor::build(kCore, "B2", {naux, maxpq});
    double* Bp = B2.data().data();

    // => Integrals <= //

    int nthread = omp_get_max_threads();    
    std::shared_ptr<SBasisSet> zero = SBasisSet::zero_basis();
    std::vector<std::shared_ptr<PotentialInt4C>> Bints;
    for (int t = 0; t < nthread; t++) {
        Bints.push_back(std::shared_ptr<PotentialInt4C>(new PotentialInt4C(auxiliary_,zero,primary_,primary_,0,a_,b_,w_)));
    }

    // ==> Master Loop <== //

    for (size_t PQtask = 0; PQtask < PQtasks.size() - 1; PQtask++) {

        // => Shell Task Indexing <= //

        size_t PQstart = PQtasks[PQtask];
        size_t PQstop  = PQtasks[PQtask + 1];
        size_t PQsize  = PQstop - PQstart; 
        size_t pqstart = pqstarts[PQtask];
        size_t pqstop  = (PQstop == shell_pairs.size() ? npq : pqstarts[PQtask+1]);
        size_t pqsize  = pqstop - pqstart;
        size_t APQsize = nAshell * PQsize;

        // => Integrals <= //
    
        #pragma omp parallel for schedule(dynamic)
        for (size_t APQtask = 0; APQtask < APQsize; APQtask++) {
            size_t A  = APQtask / PQsize;
            size_t PQ = APQtask % PQsize + PQstart;
            size_t P = shell_pairs[PQ].first;
            size_t Q = shell_pairs[PQ].second;
            int nA = auxiliary_->shell(A).nfunction();
            int nP = primary_->shell(P).nfunction();
            int nQ = primary_->shell(Q).nfunction();
            size_t oA = auxiliary_->shell(A).function_index();
            size_t opq = pqstarts[PQ] - pqstart; 
            int t = omp_get_thread_num();
            Bints[t]->compute_shell(A,0,P,Q);
            double* Bbuffer = Bints[t]->buffer();
            for (int a = 0; a < nA; a++) {
            for (int p = 0; p < nP; p++) {
            for (int q = 0; q < nQ; q++) {
                Bp[(a + oA) * maxpq + p*nQ + q + opq] = (*Bbuffer++);
            }}}
        }
      
        // => Fitting <= //

        for (size_t pqstart2 = 0L; pqstart2 < pqsize; pqstart2 += naux) {
            size_t pqsize2 = (pqstart2 + naux >= pqsize ? pqsize - pqstart2 : naux);
            T({{0,naux},{0,pqsize2}}) = B2({{0,naux},{pqstart2,pqstart2+pqsize2}});
            B2.gemm(J,T,false,false,naux,pqsize2,naux,naux,naux,maxpq,0,0,pqstart2,1.0,0.0);
        }

        // => Disk Striping <= //

        B({{0,naux},{pqstart,pqstart+pqsize}}) = B2({{0,naux},{0,pqsize}});
    }
    
    return B;
}

} // namespace lightspeed
