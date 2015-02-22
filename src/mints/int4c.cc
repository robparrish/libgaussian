#include <string.h>
#include "int4c.h"
#include "constants.h"

namespace libgaussian {

Int4C::Int4C(
    const std::shared_ptr<SBasisSet>& basis1,
    const std::shared_ptr<SBasisSet>& basis2,
    const std::shared_ptr<SBasisSet>& basis3,
    const std::shared_ptr<SBasisSet>& basis4,
    int deriv) :
    basis1_(basis1),
    basis2_(basis2),
    basis3_(basis3),
    basis4_(basis4),
    deriv_(deriv)
{
    is_spherical_ = true;
    am_info_ = SAngularMomentum::build(max_am());

    size_t nprim4 = 
        basis1_->max_nprimitive() *
        basis2_->max_nprimitive() *
        basis3_->max_nprimitive() *
        basis4_->max_nprimitive();

    init_libint_base();
    init_libint(&libint_, max_am(), nprim4);
}
Int4C::~Int4C()
{
    free_libint(&libint_);
}
int Int4C::max_am() const
{
    return std::max(
            std::max(
            basis1_->max_am(),
            basis2_->max_am()),
            std::max(
            basis3_->max_am(),
            basis4_->max_am()));
}
int Int4C::total_am() const
{
    return
        basis1_->max_am() +
        basis2_->max_am() +
        basis3_->max_am() +
        basis4_->max_am();
}
size_t Int4C::chunk_size() const
{
    return
        basis1_->max_ncartesian() *
        basis2_->max_ncartesian() *
        basis3_->max_ncartesian() *
        basis4_->max_ncartesian();
}
void Int4C::compute_shell(
    size_t shell1,
    size_t shell2,
    size_t shell3,
    size_t shell4)
{
    compute_quartet(
        basis1_->shell(shell1),
        basis2_->shell(shell2),
        basis3_->shell(shell3),
        basis4_->shell(shell4));
}
void Int4C::compute_shell1(
    size_t shell1,
    size_t shell2,
    size_t shell3,
    size_t shell4)
{
    compute_quartet1(
        basis1_->shell(shell1),
        basis2_->shell(shell2),
        basis3_->shell(shell3),
        basis4_->shell(shell4));
}
void Int4C::compute_shell2(
    size_t shell1,
    size_t shell2,
    size_t shell3,
    size_t shell4)
{
    compute_quartet2(
        basis1_->shell(shell1),
        basis2_->shell(shell2),
        basis3_->shell(shell3),
        basis4_->shell(shell4));
}
void Int4C::compute_quartet(
    const SGaussianShell& sh1,
    const SGaussianShell& sh2,
    const SGaussianShell& sh3,
    const SGaussianShell& sh4)
{
    throw std::runtime_error("Int4C: compute_quartet not implemented for this type");
}
void Int4C::compute_quartet1(
    const SGaussianShell& sh1,
    const SGaussianShell& sh2,
    const SGaussianShell& sh3,
    const SGaussianShell& sh4)
{
    throw std::runtime_error("Int4C: compute_quartet1 not implemented for this type");
}
void Int4C::compute_quartet2(
    const SGaussianShell& sh1,
    const SGaussianShell& sh2,
    const SGaussianShell& sh3,
    const SGaussianShell& sh4)
{
    throw std::runtime_error("Int4C: compute_quartet2 not implemented for this type");
}
void Int4C::apply_spherical(
    int L1,
    int L2,
    int L3,
    int L4,
    bool S1,
    bool S2,
    bool S3,
    bool S4,
    double* target,
    double* scratch)
{
    if (!S1 && !S2 && !S3 && !S4) return;
    
    int ncart1 = constants::ncartesian(L1);
    int ncart2 = constants::ncartesian(L2);
    int ncart3 = constants::ncartesian(L3);
    int ncart4 = constants::ncartesian(L4);

    int npure1 = constants::nspherical(L1);
    int npure2 = constants::nspherical(L2);
    int npure3 = constants::nspherical(L3);
    int npure4 = constants::nspherical(L4);

    int nfun1 = (S1 ? npure1 : ncart1);
    int nfun2 = (S2 ? npure2 : ncart2);
    int nfun3 = (S3 ? npure3 : ncart3);
    int nfun4 = (S4 ? npure4 : ncart4);

    if (S4 && L4 > 0) {
        memset(scratch,'\0',sizeof(double)*ncart1*ncart2*ncart3*npure4);
        const SAngularMomentum& trans = am_info_[L4];
        size_t ncoef = trans.ncoef();
        const std::vector<int>& cart_inds  = trans.cartesian_inds();
        const std::vector<int>& pure_inds  = trans.spherical_inds();
        const std::vector<double>& cart_coefs = trans.cartesian_coefs();
        for (size_t p = 0L; p < ncart1*ncart2*ncart3; p++) {
            double* cartp = target  + p * ncart4;
            double* purep = scratch + p * npure4;
            for (size_t ind = 0L; ind < ncoef; ind++) {
                *(purep + pure_inds[ind]) += cart_coefs[ind] * *(cartp + cart_inds[ind]);
            }
        } 
    } else {
        memcpy(scratch,target,sizeof(double)*ncart1*ncart2*ncart3*ncart4);
    }

    if (S3 && L3 > 0) {
        memset(target,'\0',sizeof(double)*ncart1*ncart2*npure3*nfun4);
        const SAngularMomentum& trans = am_info_[L3];
        size_t ncoef = trans.ncoef();
        const std::vector<int>& cart_inds  = trans.cartesian_inds();
        const std::vector<int>& pure_inds  = trans.spherical_inds();
        const std::vector<double>& cart_coefs = trans.cartesian_coefs();
        for (size_t p = 0L; p < ncart1*ncart2; p++) {
            double* cart2p = scratch + p * ncart3 * nfun4;
            double* pure2p = target  + p * npure3 * nfun4;
            for (size_t ind = 0L; ind < ncoef; ind++) {
                double* cartp = cart2p + cart_inds[ind] * nfun4;
                double* purep = pure2p + pure_inds[ind] * nfun4;
                double coef = cart_coefs[ind];
                for (size_t p2 = 0L; p2 < nfun4; p2++) {
                    *purep++ += coef * *cartp++;
                }
            }
        } 
    } else {
        memcpy(target,scratch,sizeof(double)*ncart1*ncart2*ncart3*nfun4);
    }

    if (S2 && L2 > 0) {
        memset(scratch,'\0',sizeof(double)*ncart1*npure2*nfun3*nfun4);
        const SAngularMomentum& trans = am_info_[L2];
        size_t ncoef = trans.ncoef();
        const std::vector<int>& cart_inds  = trans.cartesian_inds();
        const std::vector<int>& pure_inds  = trans.spherical_inds();
        const std::vector<double>& cart_coefs = trans.cartesian_coefs();
        for (size_t p = 0L; p < ncart1; p++) {
            double* cart2p = target  + p * ncart2 * nfun3 * nfun4;
            double* pure2p = scratch + p * npure2 * nfun3 * nfun4;
            for (size_t ind = 0L; ind < ncoef; ind++) {
                double* cartp = cart2p + cart_inds[ind] * nfun3 * nfun4;
                double* purep = pure2p + pure_inds[ind] * nfun3 * nfun4;
                double coef = cart_coefs[ind];
                for (size_t p2 = 0L; p2 < nfun3 * nfun4; p2++) {
                    *purep++ += coef * *cartp++;
                }
            }
        } 
    } else {
        memcpy(scratch,target,sizeof(double)*ncart1*ncart2*nfun3*nfun4);
    }

    if (S1 && L1 > 0) {
        memset(target,'\0',sizeof(double)*npure1*nfun2*nfun3*nfun4);
        const SAngularMomentum& trans = am_info_[L1];
        size_t ncoef = trans.ncoef();
        const std::vector<int>& cart_inds  = trans.cartesian_inds();
        const std::vector<int>& pure_inds  = trans.spherical_inds();
        const std::vector<double>& cart_coefs = trans.cartesian_coefs();
        for (size_t ind = 0L; ind < ncoef; ind++) {
            double* cartp = scratch + cart_inds[ind] * nfun2 * nfun3 * nfun4;
            double* purep = target  + pure_inds[ind] * nfun2 * nfun3 * nfun4;
            double coef = cart_coefs[ind];
            for (size_t p2 = 0L; p2 < nfun2 * nfun3 * nfun4; p2++) {
                *purep++ += coef * *cartp++;
            }
        }
    } else {
        memcpy(target,scratch,sizeof(double)*ncart1*nfun2*nfun3*nfun4);
    }
}

} // namespace libgaussian
