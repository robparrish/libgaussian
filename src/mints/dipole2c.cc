#include <math.h>
#include <string.h>
#include "int2c.h"
#include "constants.h"

namespace lightspeed {

DipoleInt2C::DipoleInt2C(
    const std::shared_ptr<SBasisSet>& basis1,
    const std::shared_ptr<SBasisSet>& basis2,
    int deriv) :
    Int2C(basis1,basis2,deriv),
    recursion_(basis1->max_am()+1, basis2->max_am()+1)
{
    size_t size;
    if (deriv_ == 0) {
        size = 3L * chunk_size();
    } else {
        throw std::runtime_error("DipoleInt2C: deriv too high");
    }
    data1_.resize(size);
    data2_.resize(size);
    buffer1_ = data1_.data();
    buffer2_ = data2_.data();
}
void DipoleInt2C::compute_pair(
    const SGaussianShell& sh1,
    const SGaussianShell& sh2)
{
    int ao12;
    int am1 = sh1.am();
    int am2 = sh2.am();
    int nprim1 = sh1.nprimitive();
    int nprim2 = sh2.nprimitive();
    double A[3], B[3];
    A[0] = sh1.x();
    A[1] = sh1.y();
    A[2] = sh1.z();
    B[0] = sh2.x();
    B[1] = sh2.y();
    B[2] = sh2.z();

    size_t chunk = chunk_size();

    size_t ydisp = 1L * chunk;
    size_t zdisp = 2L * chunk;

    // compute intermediates
    double AB2 = 0.0;
    AB2 += (A[0] - B[0]) * (A[0] - B[0]);
    AB2 += (A[1] - B[1]) * (A[1] - B[1]);
    AB2 += (A[2] - B[2]) * (A[2] - B[2]);

    size_t ncart2 = constants::ncartesian(am1) * constants::ncartesian(am2);
    memset(buffer1_ + 0L * chunk, '\0', sizeof(double) * ncart2);
    memset(buffer1_ + 1L * chunk, '\0', sizeof(double) * ncart2);
    memset(buffer1_ + 2L * chunk, '\0', sizeof(double) * ncart2);

    double **x = recursion_.x();
    double **y = recursion_.y();
    double **z = recursion_.z();

    for (int p1=0; p1<nprim1; ++p1) {
        double a1 = sh1.e(p1);
        double c1 = sh1.c(p1);
        for (int p2=0; p2<nprim2; ++p2) {
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

                            double x00 = x[l1][l2],   y00 = y[m1][m2],   z00 = z[n1][n2];
                            double x10 = x[l1+1][l2], y10 = y[m1+1][m2], z10 = z[n1+1][n2];

                            double DAx = (x10 + x00*(A[0]-x_)) * y00 * z00 * over_pf;
                            double DAy = x00 * (y10 + y00*(A[1]-y_)) * z00 * over_pf;
                            double DAz = x00 * y00 * (z10 + z00*(A[2]-z_)) * over_pf;

                            // Electrons have a negative charge
                            buffer1_[ao12]       -= (DAx);
                            buffer1_[ao12+ydisp] -= (DAy);
                            buffer1_[ao12+zdisp] -= (DAz);

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
        apply_spherical(am1, am2, s1, s2, buffer1_,         buffer2_);
        apply_spherical(am1, am2, s1, s2, buffer1_ + ydisp, buffer2_);
        apply_spherical(am1, am2, s1, s2, buffer1_ + zdisp, buffer2_);
    }
}
void DipoleInt2C::compute_pair1(
    const SGaussianShell& sh1,
    const SGaussianShell& sh2)
{
    throw std::runtime_error("Not Implemented");
}
void DipoleInt2C::compute_pair2(
    const SGaussianShell& sh1,
    const SGaussianShell& sh2)
{
    throw std::runtime_error("Not Implemented");
}


} // namespace lightspeed
