#include <cmath>
#include "schwarz.h"
#include "int4c.h"

#include <omp.h>

namespace lightspeed {

SchwarzSieve::SchwarzSieve(
    const std::shared_ptr<SBasisSet>& basis1,
    const std::shared_ptr<SBasisSet>& basis2,
    double cutoff,
    double a,
    double b,
    double w):
    basis1_(basis1),
    basis2_(basis2),
    cutoff_(cutoff),
    a_(a),
    b_(b),
    w_(w)
{
    build_integrals();
    build_sieve();
}
void SchwarzSieve::build_integrals()
{
    bool symm = is_symmetric();
    size_t nshell1 = basis1_->nshell();
    size_t nshell2 = basis2_->nshell();

    std::vector<std::pair<int,int> > shell_tasks;
    if (symm) {
        shell_tasks.resize(nshell1 * (nshell1 + 1) / 2);
        for (size_t P = 0, index = 0; P < nshell1; P++) {
            for (size_t Q = 0; Q <= P; Q++) {
                shell_tasks[index++] = std::pair<int,int>(P,Q);
            }
        }
    } else {
        shell_tasks.resize(nshell1 * nshell2);
        for (size_t P = 0, index = 0; P < nshell1; P++) {
            for (size_t Q = 0; Q < nshell2; Q++) {
                shell_tasks[index++] = std::pair<int,int>(P,Q);
            }
        }
    }

    // Target
    shell_maxs_.resize(nshell1*nshell2,0.0); 

    int nthread = omp_get_max_threads();
    std::vector<std::shared_ptr<PotentialInt4C>> ints;
    for (int t = 0; t < nthread; t++) {
        ints.push_back(std::shared_ptr<PotentialInt4C>(new PotentialInt4C(basis1_,basis2_,basis1_,basis2_,0,a_,b_,w_)));
    }

    #pragma omp parallel for schedule(dynamic)
    for (size_t ind = 0; ind < shell_tasks.size(); ind++) {
        int P = shell_tasks[ind].first;
        int Q = shell_tasks[ind].second;
        int nP = basis1_->shell(P).nfunction();
        int nQ = basis2_->shell(Q).nfunction();
        int t = omp_get_thread_num();
        ints[t]->compute_shell(P,Q,P,Q);
        double* buffer = ints[t]->buffer();
        double max_val = 0.0;
        for (int p = 0; p < nP; p++) {
            for (int q = 0; q < nQ; q++) {
                double I = buffer[(p*nQ + q)*nP*nQ + (p*nQ + q)];
                max_val = std::max(max_val, fabs(I));
            }
        }
        if (symm) {
            shell_maxs_[P * nshell2 + Q] = 
            shell_maxs_[Q * nshell2 + P] = 
            max_val;
        } else {
            shell_maxs_[P * nshell2 + Q] = 
            max_val;
        }    
    } 

    // Target
    overall_max_ = 0.0;
    for (double x : shell_maxs_) {
        overall_max_ = std::max(overall_max_, x);
    }
}
void SchwarzSieve::build_sieve()
{
    bool symm = is_symmetric();
    size_t nshell1 = basis1_->nshell();
    size_t nshell2 = basis2_->nshell();

    std::vector<std::pair<int,int> > shell_tasks;
    if (symm) {
        shell_tasks.resize(nshell1 * (nshell1 + 1) / 2);
        for (size_t P = 0, index = 0; P < nshell1; P++) {
            for (size_t Q = 0; Q <= P; Q++) {
                shell_tasks[index++] = std::pair<int,int>(P,Q);
            }
        }
    } else {
        shell_tasks.resize(nshell1 * nshell2);
        for (size_t P = 0, index = 0; P < nshell1; P++) {
            for (size_t Q = 0; Q < nshell2; Q++) {
                shell_tasks[index++] = std::pair<int,int>(P,Q);
            }
        }
    }

    shell_pairs_.clear();
    for (size_t ind = 0; ind < shell_tasks.size(); ind++) {
        int P = shell_tasks[ind].first;
        int Q = shell_tasks[ind].second;
        if (cutoff_ == 0.0 || sqrt(shell_maxs_[P*nshell2 + Q] * overall_max_) >= cutoff_) {
            shell_pairs_.push_back(shell_tasks[ind]);
        }
    }
}
void SchwarzSieve::print(FILE* fh) const
{
    bool symm = is_symmetric();
    size_t nshell1 = basis1_->nshell();
    size_t nshell2 = basis2_->nshell();

    size_t possible = symm ? 
        nshell1 * (nshell1 + 1) / 2 : 
        nshell1 * nshell2;
    size_t sieved = shell_pairs_.size();

    fprintf(fh, "  SchwarzSieve:\n");
    fprintf(fh, "    Cutoff             = %18.3E\n", cutoff_);
    fprintf(fh, "    Symmetric?         = %18s\n", symm ? "Yes" : "No");
    fprintf(fh, "    Basis1             = %18s\n", basis1_->name().c_str());
    fprintf(fh, "    Basis2             = %18s\n", basis2_->name().c_str());
    fprintf(fh, "    Total Shell Pairs  = %18zu\n", possible);
    fprintf(fh, "    Sieved Shell Pairs = %18zu\n", sieved);
    fprintf(fh, "    Relative Fill      = %18.3E\n", sieved / (double) possible);
    fprintf(fh, "    Maximum Integral   = %18.3E\n", overall_max_);
    fprintf(fh, "\n");
}
double SchwarzSieve::shell_estimate_PQPQ(
    size_t P,
    size_t Q) const
{
    size_t nshell2 = basis2_->nshell();
    return shell_maxs_[P*nshell2 + Q];
}
double SchwarzSieve::shell_estimate_PQRS(
    size_t P,
    size_t Q,
    size_t R,
    size_t S) const
{
    size_t nshell2 = basis2_->nshell();
    return sqrt(shell_maxs_[P*nshell2 + Q] * shell_maxs_[R*nshell2 + S]);
}


} // namespace lightspeed
