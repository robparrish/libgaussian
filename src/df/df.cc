#include <math.h>
#include <mints/int4c.h>
#include <mints/schwarz.h>
#include "df.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

using namespace ambit;

namespace lightspeed {

DFERI::DFERI(
    const std::shared_ptr<SchwarzSieve>& sieve,
    const std::shared_ptr<SBasisSet>& auxiliary) :
    sieve_(sieve),
    primary_(sieve->basis1()),
    auxiliary_(auxiliary)
{
    if (!sieve_->is_symmetric()) throw std::runtime_error("DFERI sieve must be symmetric.");

    doubles_ = 256000000L; // 1 GB
    a_ = 1.0;
    b_ = 0.0;
    w_ = 0.0;
    metric_condition_ = 1.0E-12;
}
Tensor DFERI::metric_core() const
{
    size_t nshell = auxiliary_->nshell();
    size_t naux = auxiliary_->nfunction();

    Tensor J = Tensor::build(kCore, "J", {naux, naux});
    double* Jp = J.data().data();

    std::shared_ptr<SBasisSet> zero = SBasisSet::zero_basis();

    #if defined(_OPENMP)
    int nthread = omp_get_max_threads();
    #else
    int nthread = 1;
    #endif
    std::vector<std::shared_ptr<PotentialInt4C>> Jints;
    for (int t = 0; t < nthread; t++) {
        Jints.push_back(std::shared_ptr<PotentialInt4C>(new PotentialInt4C(auxiliary_,zero,auxiliary_,zero,0,a_,b_,w_)));
    }

    std::vector<std::pair<int,int> > shell_pairs(nshell * (nshell + 1) / 2);
    for (size_t P = 0, index = 0; P < nshell; P++) {
        for (size_t Q = 0; Q <= P; Q++) {
            shell_pairs[index++] = std::pair<int,int>(P,Q);
        }
    }

    #pragma omp parallel for schedule(dynamic)
    for (size_t ind = 0; ind < shell_pairs.size(); ind++) {
        int P = shell_pairs[ind].first;
        int Q = shell_pairs[ind].second;
        int nP = auxiliary_->shell(P).nfunction();
        int nQ = auxiliary_->shell(Q).nfunction();
        int oP = auxiliary_->shell(P).function_index();
        int oQ = auxiliary_->shell(Q).function_index();
        #if defined(_OPENMP)
        int t = omp_get_thread_num();
        #else
        int t = 0;
        #endif
        Jints[t]->compute_shell(P,0,Q,0);
        double* Jbuffer = Jints[t]->buffer();
        for (int p = 0; p < nP; p++) {
            for (int q = 0; q < nQ; q++) {
                Jp[(p + oP)*naux + (q + oQ)] =
                Jp[(q + oQ)*naux + (p + oP)] =
                Jbuffer[p*nQ + q];
            }
        }
    }

    return J;
}
Tensor DFERI::metric_power_core(
    double power,
    double condition) const
{
    Tensor J = metric_core();
    return J.power(power,condition);
}


} // namespace lightspeed
