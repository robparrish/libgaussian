#include <string.h>
#include "int2c.h"
#include "constants.h"

namespace libgaussian {

Int2C::Int2C(
    const std::shared_ptr<SBasisSet>& basis1,
    const std::shared_ptr<SBasisSet>& basis2,
    int deriv) :
    basis1_(basis1),
    basis2_(basis2),
    deriv_(deriv)
{
    is_spherical_ = true;
    am_info_ = SAngularMomentum::build(max_am());
    x_ = 0.0;
    y_ = 0.0;
    z_ = 0.0;
}
int Int2C::max_am() const 
{
    return std::max(basis1_->max_am(),basis2_->max_am());
}
int Int2C::total_am() const 
{
    return basis1_->max_am() + basis2_->max_am();
}
size_t Int2C::chunk_size() const 
{
    return basis1_->max_ncartesian() * basis2_->max_ncartesian();
}
void Int2C::compute_shell(
    size_t shell1,
    size_t shell2)
{
    compute_pair(basis1_->shell(shell1),basis2_->shell(shell2));
}
void Int2C::compute_shell1(
    size_t shell1,
    size_t shell2)
{
    compute_pair1(basis1_->shell(shell1),basis2_->shell(shell2));
}
void Int2C::compute_shell2(
    size_t shell1,
    size_t shell2)
{
    compute_pair2(basis1_->shell(shell1),basis2_->shell(shell2));
}
void Int2C::compute_pair(
    const SGaussianShell& sh1,
    const SGaussianShell& sh2)
{
    throw std::runtime_error("Int2C: compute_pair not implemented for this type");
}
void Int2C::compute_pair1(
    const SGaussianShell& sh1,
    const SGaussianShell& sh2)
{
    throw std::runtime_error("Int2C: compute_pair1 not implemented for this type");
}
void Int2C::compute_pair2(
    const SGaussianShell& sh1,
    const SGaussianShell& sh2)
{
    throw std::runtime_error("Int2C: compute_pair2 not implemented for this type");
}
void Int2C::apply_spherical(
    int L1,
    int L2,
    bool S1,
    bool S2,
    double* target,
    double* scratch)
{
    if (!S1 && !S2) return;
    
    int ncart1 = constants::ncartesian(L1);
    int ncart2 = constants::ncartesian(L2);

    int npure1 = constants::nspherical(L1);
    int npure2 = constants::nspherical(L2);

    int nfun1 = (S1 ? npure1 : ncart1);
    int nfun2 = (S2 ? npure2 : ncart2);

    if (S2 && L2 > 0) {
        memset(scratch,'\0',sizeof(double)*ncart1*npure2);
        const SAngularMomentum& trans = am_info_[L2];
        size_t ncoef = trans.ncoef();
        const std::vector<int>& cart_inds  = trans.cartesian_inds();
        const std::vector<int>& pure_inds  = trans.spherical_inds();
        const std::vector<double>& cart_coefs = trans.cartesian_coefs();
        for (int p1 = 0; p1 < ncart1; p1++) {
            double* cartp = target  + p1 * ncart2;
            double* purep = scratch + p1 * npure2;
            for (size_t ind = 0L; ind < ncoef; ind++) {
                *(purep + pure_inds[ind]) += cart_coefs[ind] * *(cartp + cart_inds[ind]);
            }
        } 
    } else {
        memcpy(scratch,target,sizeof(double)*ncart1*ncart2);
    }

    if (S1 && L1 > 0) {
        memset(target,'\0',sizeof(double)*npure1*nfun2);
        const SAngularMomentum& trans = am_info_[L1];
        size_t ncoef = trans.ncoef();
        const std::vector<int>& cart_inds  = trans.cartesian_inds();
        const std::vector<int>& pure_inds  = trans.spherical_inds();
        const std::vector<double>& cart_coefs = trans.cartesian_coefs();
        for (size_t ind = 0L; ind < ncoef; ind++) {
            double* cartp = scratch + cart_inds[ind] * nfun2;
            double* purep = target  + pure_inds[ind] * nfun2;
            double coef = cart_coefs[ind];
            for (int p2 = 0; p2 < nfun2; p2++) {
                *purep++ += coef * *cartp++;
            }
        } 
    } else {
        memcpy(target,scratch,sizeof(double)*ncart1*nfun2);
    }
}

} // namespace libgaussian
