#include <math.h>
#include <string.h>
#include "int2c.h"

namespace lightspeed {

OverlapInt2C::OverlapInt2C(
    const std::shared_ptr<SBasisSet>& basis1,
    const std::shared_ptr<SBasisSet>& basis2,
    int deriv) :
    Int2C(basis1,basis2,deriv),
    recursion_(basis1->max_am()+deriv, basis2->max_am()+deriv)
{
    size_t size;
    if (deriv_ == 0) {
        size = 1L * chunk_size();
    } else if (deriv_ == 1) {
        size = 6L * chunk_size();
    } else {
        throw std::runtime_error("OverlapInt2C: deriv too high");
    }
    data1_.resize(size);
    data2_.resize(size);
    buffer1_ = data1_.data();
    buffer2_ = data2_.data();
}

void OverlapInt2C::compute_pair(
    const SGaussianShell& sh1,
    const SGaussianShell& sh2)
{
    int ao12;
    int am1 = sh1.am();
    int am2 = sh2.am();
    size_t nprim1 = sh1.nprimitive();
    size_t nprim2 = sh2.nprimitive();
    double A[3], B[3];
    A[0] = sh1.x();
    A[1] = sh1.y();
    A[2] = sh1.z();
    B[0] = sh2.x();
    B[1] = sh2.y();
    B[2] = sh2.z();

    // compute intermediates
    double AB2 = 0.0;
    AB2 += (A[0] - B[0]) * (A[0] - B[0]);
    AB2 += (A[1] - B[1]) * (A[1] - B[1]);
    AB2 += (A[2] - B[2]) * (A[2] - B[2]);

    memset(buffer1_, 0, sh1.ncartesian() * sh2.ncartesian() * sizeof(double));

    double **x = recursion_.x();
    double **y = recursion_.y();
    double **z = recursion_.z();

    for (size_t p1=0; p1<nprim1; ++p1) {
        double a1 = sh1.e(p1);
        double c1 = sh1.c(p1);
        for (size_t p2=0; p2<nprim2; ++p2) {
            double a2 = sh2.e(p2);
            double c2 = sh2.c(p2);
            double gamma = a1 + a2;
            double oog = 1.0/gamma;

            double PA[3], PB[3];
            double P[3];

            P[0] = (a1*A[0] + a2*B[0])*oog;
            P[1] = (a1*A[1] + a2*B[1])*oog;
            P[2] = (a1*A[2] + a2*B[2])*oog;
            PA[0] = P[0] - A[0];
            PA[1] = P[1] - A[1];
            PA[2] = P[2] - A[2];
            PB[0] = P[0] - B[0];
            PB[1] = P[1] - B[1];
            PB[2] = P[2] - B[2];

            double over_pf = exp(-a1*a2*AB2*oog) * sqrt(M_PI*oog) * M_PI * oog * c1 * c2;

            // Do recursion
            recursion_.compute(PA, PB, gamma, am1, am2);

            ao12 = 0;
            for(int ii = 0; ii <= am1; ii++) {
                int l1 = am1 - ii;
                for(int jj = 0; jj <= ii; jj++) {
                    int m1 = ii - jj;
                    int n1 = jj;
                    /*--- create all am components of sj ---*/
                    for(int kk = 0; kk <= am2; kk++) {
                        int l2 = am2 - kk;
                        for(int ll = 0; ll <= kk; ll++) {
                            int m2 = kk - ll;
                            int n2 = ll;

                            double x0 = x[l1][l2];
                            double y0 = y[m1][m2];
                            double z0 = z[n1][n2];

                            buffer1_[ao12++] += over_pf*x0*y0*z0;
                        }
                    }
                }
            }
        }
    }

    bool s1 = sh1.is_spherical();
    bool s2 = sh2.is_spherical();
    if (is_spherical_) apply_spherical(am1, am2, s1, s2, buffer1_, buffer2_);
}

void OverlapInt2C::compute_pair1(
    const SGaussianShell& sh1,
    const SGaussianShell& sh2)
{
    int ao12;
    const int am1 = sh1.am();
    const int am2 = sh2.am();
    const size_t nprim1 = sh1.nprimitive();
    const size_t nprim2 = sh2.nprimitive();

    double A[3], B[3];
    A[0] = sh1.x();
    A[1] = sh1.y();
    A[2] = sh1.z();
    B[0] = sh2.x();
    B[1] = sh2.y();
    B[2] = sh2.z();

    // size of the length of a perturbation
    const size_t size = sh1.ncartesian() * sh2.ncartesian();
    const size_t center_i_start = 0;       // always 0
    const size_t center_j_start = 3*size;  // skip over x, y, z of center i

    // compute intermediates
    double AB2 = 0.0;
    AB2 += (A[0] - B[0]) * (A[0] - B[0]);
    AB2 += (A[1] - B[1]) * (A[1] - B[1]);
    AB2 += (A[2] - B[2]) * (A[2] - B[2]);

    memset(buffer1_, 0, 6L * size * sizeof(double));

    double **x = recursion_.x();
    double **y = recursion_.y();
    double **z = recursion_.z();

    for (size_t p1=0; p1<nprim1; ++p1) {
        const double a1 = sh1.e(p1);
        const double c1 = sh1.c(p1);
        for (size_t p2=0; p2<nprim2; ++p2) {
            const double a2 = sh2.e(p2);
            const double c2 = sh2.c(p2);
            const double gamma = a1 + a2;
            const double oog = 1.0/gamma;

            double PA[3], PB[3];
            double P[3];

            P[0] = (a1*A[0] + a2*B[0])*oog;
            P[1] = (a1*A[1] + a2*B[1])*oog;
            P[2] = (a1*A[2] + a2*B[2])*oog;
            PA[0] = P[0] - A[0];
            PA[1] = P[1] - A[1];
            PA[2] = P[2] - A[2];
            PB[0] = P[0] - B[0];
            PB[1] = P[1] - B[1];
            PB[2] = P[2] - B[2];

            const double over_pf = exp(-a1*a2*AB2*oog) * sqrt(M_PI*oog) * M_PI * oog * c1 * c2;

            recursion_.compute(PA, PB, gamma, am1+1, am2+1);

            ao12 = 0;
            for(int ii = 0; ii <= am1; ii++) {
                const int l1 = am1 - ii;
                for(int jj = 0; jj <= ii; jj++) {
                    const int m1 = ii - jj;
                    const int n1 = jj;

                    for(int kk = 0; kk <= am2; kk++) {
                        const int l2 = am2 - kk;
                        for(int ll = 0; ll <= kk; ll++) {
                            const int m2 = kk - ll;
                            const int n2 = ll;

                            double ix = 0.0, iy = 0.0, iz = 0.0;

                            //Using Translational Invariance
                            // x on center i
                            ix += 2.0*a1*over_pf*x[l1+1][l2]*y[m1][m2]*z[n1][n2];
                            if (l1)
                                ix -= l1*over_pf*x[l1-1][l2]*y[m1][m2]*z[n1][n2];
                            // y on center i
                            iy += 2.0*a1*over_pf*x[l1][l2]*y[m1+1][m2]*z[n1][n2];
                            if (m1)
                                iy -= m1*over_pf*x[l1][l2]*y[m1-1][m2]*z[n1][n2];
                            // z on center i
                            iz += 2.0*a1*over_pf*x[l1][l2]*y[m1][m2]*z[n1+1][n2];
                            if (n1)
                                iz -= n1*over_pf*x[l1][l2]*y[m1][m2]*z[n1-1][n2];

                            // x on center i,j
                            buffer1_[center_i_start+(0*size)+ao12] += ix;
                            buffer1_[center_j_start+(0*size)+ao12] -= ix;
                            // y on center i,j
                            buffer1_[center_i_start+(1*size)+ao12] += iy;
                            buffer1_[center_j_start+(1*size)+ao12] -= iy;
                            // z on center i,j
                            buffer1_[center_i_start+(2*size)+ao12] += iz;
                            buffer1_[center_j_start+(2*size)+ao12] -= iz;

                            ao12++;
                        }
                    }
                }
            }
        }
    }

    bool s1 = sh1.is_spherical();
    bool s2 = sh2.is_spherical();
    if (is_spherical_) {
        apply_spherical(am1, am2, s1, s2, buffer1_ + 0L * chunk_size(), buffer2_ + 0L * chunk_size());
        apply_spherical(am1, am2, s1, s2, buffer1_ + 1L * chunk_size(), buffer2_ + 1L * chunk_size());
        apply_spherical(am1, am2, s1, s2, buffer1_ + 2L * chunk_size(), buffer2_ + 2L * chunk_size());
        apply_spherical(am1, am2, s1, s2, buffer1_ + 3L * chunk_size(), buffer2_ + 3L * chunk_size());
        apply_spherical(am1, am2, s1, s2, buffer1_ + 4L * chunk_size(), buffer2_ + 4L * chunk_size());
        apply_spherical(am1, am2, s1, s2, buffer1_ + 5L * chunk_size(), buffer2_ + 5L * chunk_size());
    }
}

void OverlapInt2C::compute_pair2(
    const SGaussianShell& sh1,
    const SGaussianShell& sh2)
{
    throw std::runtime_error("Not Implemented");
}


} // namespace lightspeed
