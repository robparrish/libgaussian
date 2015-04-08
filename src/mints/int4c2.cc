#include "int4c2.h"
#include "interaction.h"
#include "constants.h"
#include <core/basisset.h>
#include <core/am.h>
#include <string.h>

namespace lightspeed {

namespace {

void permute_1234_to_1243(double *s, double *t, size_t nbf1, size_t nbf2, size_t nbf3, size_t nbf4)
{
    size_t f1 = nbf1;
    size_t f2 = nbf2;
    size_t f3 = nbf4;
    size_t f4 = nbf3;
    for (size_t bf1 = 0; bf1 < f1; bf1++) {
        for (size_t bf2 = 0; bf2 < f2; bf2++) {
            for (size_t bf4 = 0; bf4 < f4; bf4++) {
                for (size_t bf3 = 0; bf3 < f3; bf3++) {
                    double *t_ptr = t + ((bf1 * f2 + bf2) * f3 + bf3) * f4 + bf4;
                    *(t_ptr) = *(s++);
                }
            }
        }
    }
}

void permute_1234_to_2134(double *s, double *t, size_t nbf1, size_t nbf2, size_t nbf3, size_t nbf4)
{
    size_t f1 = nbf2;
    size_t f2 = nbf1;
    size_t f3 = nbf3;
    size_t f4 = nbf4;
    for (size_t bf2 = 0; bf2 < f2; bf2++) {
        for (size_t bf1 = 0; bf1 < f1; bf1++) {
            for (size_t bf3 = 0; bf3 < f3; bf3++) {
                for (size_t bf4 = 0; bf4 < f4; bf4++) {
                    double *t_ptr = t + ((bf1 * f2 + bf2) * f3 + bf3) * f4 + bf4;
                    *(t_ptr) = *(s++);
                }
            }
        }
    }
}

void permute_1234_to_2143(double *s, double *t, size_t nbf1, size_t nbf2, size_t nbf3, size_t nbf4)
{
    size_t f1 = nbf2;
    size_t f2 = nbf1;
    size_t f3 = nbf4;
    size_t f4 = nbf3;
    for (size_t bf2 = 0; bf2 < f2; bf2++) {
        for (size_t bf1 = 0; bf1 < f1; bf1++) {
            for (size_t bf4 = 0; bf4 < f4; bf4++) {
                for (size_t bf3 = 0; bf3 < f3; bf3++) {
                    double *t_ptr = t + ((bf1 * f2 + bf2) * f3 + bf3) * f4 + bf4;
                    *(t_ptr) = *(s++);
                }
            }
        }
    }
}

void permute_1234_to_3412(double *s, double *t, size_t nbf1, size_t nbf2, size_t nbf3, size_t nbf4)
{
    size_t f1 = nbf3;
    size_t f2 = nbf4;
    size_t f3 = nbf1;
    size_t f4 = nbf2;
    for (size_t bf3 = 0; bf3 < f3; bf3++) {
        for (size_t bf4 = 0; bf4 < f4; bf4++) {
            for (size_t bf1 = 0; bf1 < f1; bf1++) {
                for (size_t bf2 = 0; bf2 < f2; bf2++) {
                    double *t_ptr = t + ((bf1 * f2 + bf2) * f3 + bf3) * f4 + bf4;
                    *(t_ptr) = *(s++);
                }
            }
        }
    }
}

void permute_1234_to_4312(double *s, double *t, size_t nbf1, size_t nbf2, size_t nbf3, size_t nbf4)
{
    size_t f1 = nbf4;
    size_t f2 = nbf3;
    size_t f3 = nbf1;
    size_t f4 = nbf2;
    for (size_t bf3 = 0; bf3 < f3; bf3++) {
        for (size_t bf4 = 0; bf4 < f4; bf4++) {
            for (size_t bf2 = 0; bf2 < f2; bf2++) {
                for (size_t bf1 = 0; bf1 < f1; bf1++) {
                    double *t_ptr = t + ((bf1 * f2 + bf2) * f3 + bf3) * f4 + bf4;
                    *(t_ptr) = *(s++);
                }
            }
        }
    }
}

void permute_1234_to_3421(double *s, double *t, size_t nbf1, size_t nbf2, size_t nbf3, size_t nbf4)
{
    size_t f1 = nbf3;
    size_t f2 = nbf4;
    size_t f3 = nbf2;
    size_t f4 = nbf1;
    for (size_t bf4 = 0; bf4 < f4; bf4++) {
        for (size_t bf3 = 0; bf3 < f3; bf3++) {
            for (size_t bf1 = 0; bf1 < f1; bf1++) {
                for (size_t bf2 = 0; bf2 < f2; bf2++) {
                    double *t_ptr = t + ((bf1 * f2 + bf2) * f3 + bf3) * f4 + bf4;
                    *(t_ptr) = *(s++);
                }
            }
        }
    }
}

void permute_1234_to_4321(double *s, double *t, size_t nbf1, size_t nbf2, size_t nbf3, size_t nbf4)
{
    size_t f1 = nbf4;
    size_t f2 = nbf3;
    size_t f3 = nbf2;
    size_t f4 = nbf1;
    for (size_t bf4 = 0; bf4 < f4; bf4++) {
        for (size_t bf3 = 0; bf3 < f3; bf3++) {
            for (size_t bf2 = 0; bf2 < f2; bf2++) {
                for (size_t bf1 = 0; bf1 < f1; bf1++) {
                    double *t_ptr = t + ((bf1 * f2 + bf2) * f3 + bf3) * f4 + bf4;
                    *(t_ptr) = *(s++);
                }
            }
        }
    }
}

void permute_target(double *s, double *t,
                    const SGaussianShell *s1,
                    const SGaussianShell *s2,
                    const SGaussianShell *s3,
                    const SGaussianShell *s4,
                    bool p12, bool p34, bool p13p24)
{
    size_t nbf1, nbf2, nbf3, nbf4;

    nbf1 = s1->nfunction();
    nbf2 = s2->nfunction();
    nbf3 = s3->nfunction();
    nbf4 = s4->nfunction();

    if (!p13p24) {
        if (p12) {
            if (p34) {
                permute_1234_to_2143(s, t, nbf1, nbf2, nbf3, nbf4);
            } else {
                permute_1234_to_2134(s, t, nbf1, nbf2, nbf3, nbf4);
            }
        } else {
            permute_1234_to_1243(s, t, nbf1, nbf2, nbf3, nbf4);
        }
    } else {
        if (p12) {
            if (p34) {
                permute_1234_to_4321(s, t, nbf1, nbf2, nbf3, nbf4);
            } else {
                permute_1234_to_4312(s, t, nbf1, nbf2, nbf3, nbf4);
            }
        } else {
            if (p34) {
                permute_1234_to_3421(s, t, nbf1, nbf2, nbf3, nbf4);
            } else {
                permute_1234_to_3412(s, t, nbf1, nbf2, nbf3, nbf4);
            }
        }
    }
}

} // namespace anonymous

Int4C2::Int4C2(
        const std::shared_ptr<SBasisSet> &basis1,
        const std::shared_ptr<SBasisSet> &basis2,
        const std::shared_ptr<SBasisSet> &basis3,
        const std::shared_ptr<SBasisSet> &basis4,
        const std::shared_ptr<Interaction> &interaction,
        int deriv) :
        basis1_(basis1),
        basis2_(basis2),
        basis3_(basis3),
        basis4_(basis4),
        interaction_(interaction),
        deriv_(deriv)
{
    is_spherical_ = true;
    am_info_ = SAngularMomentum::build(max_am());
}

std::shared_ptr<Int4C2> Int4C2::coulomb(
    const std::shared_ptr<SBasisSet> &basis1,
    const std::shared_ptr<SBasisSet> &basis2,
    const std::shared_ptr<SBasisSet> &basis3,
    const std::shared_ptr<SBasisSet> &basis4,
    int deriv,
    const std::string& type) 
{
    return std::shared_ptr<Int4C2>(new Int4C2(basis1,basis2,basis3,basis4,Interaction::coulomb(),deriv));
}
std::shared_ptr<Int4C2> Int4C2::ewald(
    const std::shared_ptr<SBasisSet> &basis1,
    const std::shared_ptr<SBasisSet> &basis2,
    const std::shared_ptr<SBasisSet> &basis3,
    const std::shared_ptr<SBasisSet> &basis4,
    int deriv,
    double a,
    double b,
    double w,
    const std::string& type) 
{
    return std::shared_ptr<Int4C2>(new Int4C2(basis1,basis2,basis3,basis4,Interaction::ewald(a,b,w),deriv));
}

int Int4C2::max_am() const
{
    return std::max(
            std::max(
                    basis1_->max_am(),
                    basis2_->max_am()),
            std::max(
                    basis3_->max_am(),
                    basis4_->max_am()));
}
int Int4C2::total_am() const
{
    return
            basis1_->max_am() +
                    basis2_->max_am() +
                    basis3_->max_am() +
                    basis4_->max_am();
}
size_t Int4C2::chunk_size() const
{
    return
            basis1_->max_ncartesian() *
                    basis2_->max_ncartesian() *
                    basis3_->max_ncartesian() *
                    basis4_->max_ncartesian();
}
void Int4C2::compute_shell(
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
void Int4C2::compute_shell1(
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
void Int4C2::compute_shell2(
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
void Int4C2::compute_quartet(
        const SGaussianShell &sh1,
        const SGaussianShell &sh2,
        const SGaussianShell &sh3,
        const SGaussianShell &sh4)
{
    throw std::runtime_error("Int4C2: compute_quartet not implemented for this type");
}
void Int4C2::compute_quartet1(
        const SGaussianShell &sh1,
        const SGaussianShell &sh2,
        const SGaussianShell &sh3,
        const SGaussianShell &sh4)
{
    throw std::runtime_error("Int4C2: compute_quartet1 not implemented for this type");
}
void Int4C2::compute_quartet2(
        const SGaussianShell &sh1,
        const SGaussianShell &sh2,
        const SGaussianShell &sh3,
        const SGaussianShell &sh4)
{
    throw std::runtime_error("Int4C2: compute_quartet2 not implemented for this type");
}
void Int4C2::apply_spherical(
        int L1,
        int L2,
        int L3,
        int L4,
        bool S1,
        bool S2,
        bool S3,
        bool S4,
        double *target,
        double *scratch)
{
    if (!S1 && !S2 && !S3 && !S4) return;

    size_t ncart1 = constants::ncartesian(L1);
    size_t ncart2 = constants::ncartesian(L2);
    size_t ncart3 = constants::ncartesian(L3);
    size_t ncart4 = constants::ncartesian(L4);

    size_t npure1 = constants::nspherical(L1);
    size_t npure2 = constants::nspherical(L2);
    size_t npure3 = constants::nspherical(L3);
    size_t npure4 = constants::nspherical(L4);

    size_t nfun1 = (S1 ? npure1 : ncart1);
    size_t nfun2 = (S2 ? npure2 : ncart2);
    size_t nfun3 = (S3 ? npure3 : ncart3);
    size_t nfun4 = (S4 ? npure4 : ncart4);

    if (S4 && L4 > 0) {
        memset(scratch, '\0', sizeof(double) * ncart1 * ncart2 * ncart3 * npure4);
        const SAngularMomentum &trans = am_info_[L4];
        size_t ncoef = trans.ncoef();
        const std::vector<int> &cart_inds = trans.cartesian_inds();
        const std::vector<int> &pure_inds = trans.spherical_inds();
        const std::vector<double> &cart_coefs = trans.cartesian_coefs();
        for (size_t p = 0L; p < ncart1 * ncart2 * ncart3; p++) {
            double *cartp = target + p * ncart4;
            double *purep = scratch + p * npure4;
            for (size_t ind = 0L; ind < ncoef; ind++) {
                *(purep + pure_inds[ind]) += cart_coefs[ind] * *(cartp + cart_inds[ind]);
            }
        }
    } else {
        memcpy(scratch, target, sizeof(double) * ncart1 * ncart2 * ncart3 * ncart4);
    }

    if (S3 && L3 > 0) {
        memset(target, '\0', sizeof(double) * ncart1 * ncart2 * npure3 * nfun4);
        const SAngularMomentum &trans = am_info_[L3];
        size_t ncoef = trans.ncoef();
        const std::vector<int> &cart_inds = trans.cartesian_inds();
        const std::vector<int> &pure_inds = trans.spherical_inds();
        const std::vector<double> &cart_coefs = trans.cartesian_coefs();
        for (size_t p = 0L; p < ncart1 * ncart2; p++) {
            double *cart2p = scratch + p * ncart3 * nfun4;
            double *pure2p = target + p * npure3 * nfun4;
            for (size_t ind = 0L; ind < ncoef; ind++) {
                double *cartp = cart2p + cart_inds[ind] * nfun4;
                double *purep = pure2p + pure_inds[ind] * nfun4;
                double coef = cart_coefs[ind];
                for (size_t p2 = 0L; p2 < nfun4; p2++) {
                    *purep++ += coef * *cartp++;
                }
            }
        }
    } else {
        memcpy(target, scratch, sizeof(double) * ncart1 * ncart2 * ncart3 * nfun4);
    }

    if (S2 && L2 > 0) {
        memset(scratch, '\0', sizeof(double) * ncart1 * npure2 * nfun3 * nfun4);
        const SAngularMomentum &trans = am_info_[L2];
        size_t ncoef = trans.ncoef();
        const std::vector<int> &cart_inds = trans.cartesian_inds();
        const std::vector<int> &pure_inds = trans.spherical_inds();
        const std::vector<double> &cart_coefs = trans.cartesian_coefs();
        for (size_t p = 0L; p < ncart1; p++) {
            double *cart2p = target + p * ncart2 * nfun3 * nfun4;
            double *pure2p = scratch + p * npure2 * nfun3 * nfun4;
            for (size_t ind = 0L; ind < ncoef; ind++) {
                double *cartp = cart2p + cart_inds[ind] * nfun3 * nfun4;
                double *purep = pure2p + pure_inds[ind] * nfun3 * nfun4;
                double coef = cart_coefs[ind];
                for (size_t p2 = 0L; p2 < nfun3 * nfun4; p2++) {
                    *purep++ += coef * *cartp++;
                }
            }
        }
    } else {
        memcpy(scratch, target, sizeof(double) * ncart1 * ncart2 * nfun3 * nfun4);
    }

    if (S1 && L1 > 0) {
        memset(target, '\0', sizeof(double) * npure1 * nfun2 * nfun3 * nfun4);
        const SAngularMomentum &trans = am_info_[L1];
        size_t ncoef = trans.ncoef();
        const std::vector<int> &cart_inds = trans.cartesian_inds();
        const std::vector<int> &pure_inds = trans.spherical_inds();
        const std::vector<double> &cart_coefs = trans.cartesian_coefs();
        for (size_t ind = 0L; ind < ncoef; ind++) {
            double *cartp = scratch + cart_inds[ind] * nfun2 * nfun3 * nfun4;
            double *purep = target + pure_inds[ind] * nfun2 * nfun3 * nfun4;
            double coef = cart_coefs[ind];
            for (size_t p2 = 0L; p2 < nfun2 * nfun3 * nfun4; p2++) {
                *purep++ += coef * *cartp++;
            }
        }
    } else {
        memcpy(target, scratch, sizeof(double) * ncart1 * nfun2 * nfun3 * nfun4);
    }
}

} // namespace lightspeed
