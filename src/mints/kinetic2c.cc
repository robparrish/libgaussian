#include <math.h>
#include <string.h>
#include "int2c.h"

namespace lightspeed {

namespace {

static double ke_int(double **x, double **y, double **z, double a1, int l1, int m1, int n1,
                     double a2, int l2, int m2, int n2)
{
    double I1, I2, I3, I4;

    I1 = (l1 == 0 || l2 == 0) ? 0.0 : x[l1-1][l2-1] * y[m1][m2] * z[n1][n2];
    I2 = x[l1+1][l2+1] * y[m1][m2] * z[n1][n2];
    I3 = (l2 == 0) ? 0.0 : x[l1+1][l2-1] * y[m1][m2] * z[n1][n2];
    I4 = (l1 == 0) ? 0.0 : x[l1-1][l2+1] * y[m1][m2] * z[n1][n2];
    double Ix = 0.5 * l1 * l2 * I1 + 2.0 * a1 * a2 * I2 - a1 * l2 * I3 - l1 * a2 * I4;

    I1 = (m1 == 0 || m2 == 0) ? 0.0 : x[l1][l2] * y[m1-1][m2-1] * z[n1][n2];
    I2 = x[l1][l2] * y[m1+1][m2+1] * z[n1][n2];
    I3 = (m2 == 0) ? 0.0 : x[l1][l2] * y[m1+1][m2-1] * z[n1][n2];
    I4 = (m1 == 0) ? 0.0 : x[l1][l2] * y[m1-1][m2+1] * z[n1][n2];
    double Iy = 0.5 * m1 * m2 * I1 + 2.0 * a1 * a2 * I2 - a1 * m2 * I3 - m1 * a2 * I4;

    I1 = (n1 == 0 || n2 == 0) ? 0.0 : x[l1][l2] * y[m1][m2] * z[n1-1][n2-1];
    I2 = x[l1][l2] * y[m1][m2] * z[n1+1][n2+1];
    I3 = (n2 == 0) ? 0.0 : x[l1][l2] * y[m1][m2] * z[n1+1][n2-1];
    I4 = (n1 == 0) ? 0.0 : x[l1][l2] * y[m1][m2] * z[n1-1][n2+1];
    double Iz = 0.5 * n1 * n2 * I1 + 2.0 * a1 * a2 * I2 - a1 * n2 * I3 - n1 * a2 * I4;

    return (Ix + Iy + Iz);
}

}

KineticInt2C::KineticInt2C(
    const std::shared_ptr<SBasisSet>& basis1,
    const std::shared_ptr<SBasisSet>& basis2,
    int deriv) :
    Int2C(basis1,basis2,deriv),
    recursion_(basis1->max_am()+deriv+1, basis2->max_am()+deriv+1)
{
    size_t size;
    if (deriv_ == 0) {
        size = 1L * chunk_size();
    } else if (deriv_ == 1) {
        size = 6L * chunk_size();
    } else {
        throw std::runtime_error("KineticInt2C: deriv too high");
    }
    data1_.resize(size);
    data2_.resize(size);
    buffer1_ = data1_.data();
    buffer2_ = data2_.data();
}
void KineticInt2C::compute_pair(
    const SGaussianShell& sh1,
    const SGaussianShell& sh2)
{
    int ao12;
    const int am1 = sh1.am();
    const int am2 = sh2.am();
    const size_t nprim1 = sh1.nprimitive();
    const size_t nprim2 = sh2.nprimitive();
    const double Ax = sh1.x();
    const double Ay = sh1.y();
    const double Az = sh1.z();
    const double Bx = sh2.x();
    const double By = sh2.y();
    const double Bz = sh2.z();

    // compute intermediates
    double AB2 = 0.0;
    AB2 += (Ax - Bx) * (Ax - Bx);
    AB2 += (Ay - By) * (Ay - By);
    AB2 += (Az - Bz) * (Az - Bz);

    memset(buffer1_, 0, chunk_size() * sizeof(double));

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

            const double Px = (a1*Ax + a2*Bx)*oog;
            const double Py = (a1*Ay + a2*By)*oog;
            const double Pz = (a1*Az + a2*Bz)*oog;
            PA[0] = Px - Ax;
            PA[1] = Py - Ay;
            PA[2] = Pz - Az;
            PB[0] = Px - Bx;
            PB[1] = Py - By;
            PB[2] = Pz - Bz;

            double over_pf = exp(-a1*a2*AB2*oog) * sqrt(M_PI*oog) * M_PI * oog * c1 * c2;

            // Do recursion
            recursion_.compute(PA, PB, gamma, am1+1, am2+1);

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

                            double I1, I2, I3, I4;

                            I1 = (l1 == 0 || l2 == 0) ? 0.0 : x[l1-1][l2-1] * y[m1][m2] * z[n1][n2] * over_pf;
                            I2 = x[l1+1][l2+1] * y[m1][m2] * z[n1][n2] * over_pf;
                            I3 = (l2 == 0) ? 0.0 : x[l1+1][l2-1] * y[m1][m2] * z[n1][n2] * over_pf;
                            I4 = (l1 == 0) ? 0.0 : x[l1-1][l2+1] * y[m1][m2] * z[n1][n2] * over_pf;
                            double Ix = 0.5 * l1 * l2 * I1 + 2.0 * a1 * a2 * I2 - a1 * l2 * I3 - l1 * a2 * I4;

                            I1 = (m1 == 0 || m2 == 0) ? 0.0 : x[l1][l2] * y[m1-1][m2-1] * z[n1][n2] * over_pf;
                            I2 = x[l1][l2] * y[m1+1][m2+1] * z[n1][n2] * over_pf;
                            I3 = (m2 == 0) ? 0.0 : x[l1][l2] * y[m1+1][m2-1] * z[n1][n2] * over_pf;
                            I4 = (m1 == 0) ? 0.0 : x[l1][l2] * y[m1-1][m2+1] * z[n1][n2] * over_pf;
                            double Iy = 0.5 * m1 * m2 * I1 + 2.0 * a1 * a2 * I2 - a1 * m2 * I3 - m1 * a2 * I4;

                            I1 = (n1 == 0 || n2 == 0) ? 0.0 : x[l1][l2] * y[m1][m2] * z[n1-1][n2-1] * over_pf;
                            I2 = x[l1][l2] * y[m1][m2] * z[n1+1][n2+1] * over_pf;
                            I3 = (n2 == 0) ? 0.0 : x[l1][l2] * y[m1][m2] * z[n1+1][n2-1] * over_pf;
                            I4 = (n1 == 0) ? 0.0 : x[l1][l2] * y[m1][m2] * z[n1-1][n2+1] * over_pf;
                            double Iz = 0.5 * n1 * n2 * I1 + 2.0 * a1 * a2 * I2 - a1 * n2 * I3 - n1 * a2 * I4;

                            buffer1_[ao12++] += (Ix + Iy + Iz);
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

void KineticInt2C::compute_pair1(
    const SGaussianShell& sh1,
    const SGaussianShell& sh2)
{
    int ao12;
    const int am1 = sh1.am();
    const int am2 = sh2.am();
    const size_t nprim1 = sh1.nprimitive();
    const size_t nprim2 = sh2.nprimitive();

    const double Ax = sh1.x();
    const double Ay = sh1.y();
    const double Az = sh1.z();
    const double Bx = sh2.x();
    const double By = sh2.y();
    const double Bz = sh2.z();

    // size of the length of a perturbation
    const size_t size = chunk_size();
    const size_t center_i_start = 0;       // always 0
    const size_t center_j_start = 3*size;  // skip over x, y, z of center i

    // compute intermediates
    double AB2 = 0.0;
    AB2 += (Ax - Bx) * (Ax - Bx);
    AB2 += (Ay - By) * (Ay - By);
    AB2 += (Az - Bz) * (Az - Bz);

    memset(buffer1_, 0, 6 * size * sizeof(double));

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

            const double Px = (a1*Ax + a2*Bx)*oog;
            const double Py = (a1*Ay + a2*By)*oog;
            const double Pz = (a1*Az + a2*Bz)*oog;
            PA[0] = Px - Ax;
            PA[1] = Py - Ay;
            PA[2] = Pz - Az;
            PB[0] = Px - Bx;
            PB[1] = Py - By;
            PB[2] = Pz - Bz;

            double over_pf = exp(-a1*a2*AB2*oog) * sqrt(M_PI*oog) * M_PI * oog * c1 * c2;

            // Do recursion
            recursion_.compute(PA, PB, gamma, am1+2, am2+2);

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

                            double ix=0.0,iy=0.0,iz=0.0;
                            // x on center i
                            ix += 2.0 * a1 * ke_int(x, y, z, a1, l1+1, m1, n1, a2, l2, m2, n2) * over_pf;
                            if (l1)
                                ix -= l1 * ke_int(x, y, z, a1, l1-1, m1, n1, a2, l2, m2, n2) * over_pf;
                            // y on center i
                            iy += 2.0 * a1 * ke_int(x, y, z, a1, l1, m1+1, n1, a2, l2, m2, n2) * over_pf;
                            if (m1)
                                iy -= m1 * ke_int(x, y, z, a1, l1, m1-1, n1, a2, l2, m2, n2) * over_pf;
                            // z on center i
                            iz += 2.0 * a1 * ke_int(x, y, z, a1, l1, m1, n1+1, a2, l2, m2, n2) * over_pf;
                            if (n1)
                                iz -= n1 * ke_int(x, y, z, a1, l1, m1, n1-1, a2, l2, m2, n2) * over_pf;
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

void KineticInt2C::compute_pair2(
    const SGaussianShell& sh1,
    const SGaussianShell& sh2)
{
    throw std::runtime_error("Not Implemented");
}


} // namespace lightspeed
