#include <math.h>
#include <string.h>
#include "int4c.h"
#include "constants.h"
#include "fjt.h"

namespace libgaussian {

namespace {

void permute_1234_to_1243(double *s, double *t, int nbf1, int nbf2, int nbf3, int nbf4)
{
    int f1 = nbf1;
    int f2 = nbf2;
    int f3 = nbf4;
    int f4 = nbf3;
    for (int bf1 = 0; bf1 < f1; bf1++) {
        for (int bf2 = 0; bf2 < f2; bf2++) {
            for (int bf4 = 0; bf4 < f4; bf4++) {
                for (int bf3 = 0; bf3 < f3; bf3++) {
                    double *t_ptr = t + ((bf1 * f2 + bf2) * f3 + bf3) * f4 + bf4;
                    *(t_ptr) = *(s++);
                }
            }
        }
    }
}

void permute_1234_to_2134(double *s, double *t, int nbf1, int nbf2, int nbf3, int nbf4)
{
    int f1 = nbf2;
    int f2 = nbf1;
    int f3 = nbf3;
    int f4 = nbf4;
    for (int bf2 = 0; bf2 < f2; bf2++) {
        for (int bf1 = 0; bf1 < f1; bf1++) {
            for (int bf3 = 0; bf3 < f3; bf3++) {
                for (int bf4 = 0; bf4 < f4; bf4++) {
                    double *t_ptr = t + ((bf1 * f2 + bf2) * f3 + bf3) * f4 + bf4;
                    *(t_ptr) = *(s++);
                }
            }
        }
    }
}

void permute_1234_to_2143(double *s, double *t, int nbf1, int nbf2, int nbf3, int nbf4)
{
    int f1 = nbf2;
    int f2 = nbf1;
    int f3 = nbf4;
    int f4 = nbf3;
    for (int bf2 = 0; bf2 < f2; bf2++) {
        for (int bf1 = 0; bf1 < f1; bf1++) {
            for (int bf4 = 0; bf4 < f4; bf4++) {
                for (int bf3 = 0; bf3 < f3; bf3++) {
                    double *t_ptr = t + ((bf1 * f2 + bf2) * f3 + bf3) * f4 + bf4;
                    *(t_ptr) = *(s++);
                }
            }
        }
    }
}

void permute_1234_to_3412(double *s, double *t, int nbf1, int nbf2, int nbf3, int nbf4)
{
    int f1 = nbf3;
    int f2 = nbf4;
    int f3 = nbf1;
    int f4 = nbf2;
    for (int bf3 = 0; bf3 < f3; bf3++) {
        for (int bf4 = 0; bf4 < f4; bf4++) {
            for (int bf1 = 0; bf1 < f1; bf1++) {
                for (int bf2 = 0; bf2 < f2; bf2++) {
                    double *t_ptr = t + ((bf1 * f2 + bf2) * f3 + bf3) * f4 + bf4;
                    *(t_ptr) = *(s++);
                }
            }
        }
    }
}

void permute_1234_to_4312(double *s, double *t, int nbf1, int nbf2, int nbf3, int nbf4)
{
    int f1 = nbf4;
    int f2 = nbf3;
    int f3 = nbf1;
    int f4 = nbf2;
    for (int bf3 = 0; bf3 < f3; bf3++) {
        for (int bf4 = 0; bf4 < f4; bf4++) {
            for (int bf2 = 0; bf2 < f2; bf2++) {
                for (int bf1 = 0; bf1 < f1; bf1++) {
                    double *t_ptr = t + ((bf1 * f2 + bf2) * f3 + bf3) * f4 + bf4;
                    *(t_ptr) = *(s++);
                }
            }
        }
    }
}

void permute_1234_to_3421(double *s, double *t, int nbf1, int nbf2, int nbf3, int nbf4)
{
    int f1 = nbf3;
    int f2 = nbf4;
    int f3 = nbf2;
    int f4 = nbf1;
    for (int bf4 = 0; bf4 < f4; bf4++) {
        for (int bf3 = 0; bf3 < f3; bf3++) {
            for (int bf1 = 0; bf1 < f1; bf1++) {
                for (int bf2 = 0; bf2 < f2; bf2++) {
                    double *t_ptr = t + ((bf1 * f2 + bf2) * f3 + bf3) * f4 + bf4;
                    *(t_ptr) = *(s++);
                }
            }
        }
    }
}

void permute_1234_to_4321(double *s, double *t, int nbf1, int nbf2, int nbf3, int nbf4)
{
    int f1 = nbf4;
    int f2 = nbf3;
    int f3 = nbf2;
    int f4 = nbf1;
    for (int bf4 = 0; bf4 < f4; bf4++) {
        for (int bf3 = 0; bf3 < f3; bf3++) {
            for (int bf2 = 0; bf2 < f2; bf2++) {
                for (int bf1 = 0; bf1 < f1; bf1++) {
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
    int nbf1, nbf2, nbf3, nbf4;

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

}

PotentialInt4C::PotentialInt4C(
        const std::shared_ptr<SBasisSet> &basis1,
        const std::shared_ptr<SBasisSet> &basis2,
        const std::shared_ptr<SBasisSet> &basis3,
        const std::shared_ptr<SBasisSet> &basis4,
        int deriv,
        double a,
        double b,
        double w) :
        Int4C(basis1, basis2, basis3, basis4, deriv),
        a_(a),
        b_(b),
        w_(w)
{
    if (a != 1.0 || b != 0.0 || w != 0.0)
        throw std::logic_error("Only standard repulsion integrals are supported.");

    size_t size;
    if (deriv_ == 0) {
        size = 1L * chunk_size();
    } else {
        throw std::runtime_error("PotentialInt4C: deriv too high");
    }
    buffer1_ = new double[size];
    buffer2_ = new double[size];

    fjt_ = new FJT(basis1->max_am()+basis2->max_am()+basis3->max_am()+basis4->max_am());
}
PotentialInt4C::~PotentialInt4C()
{
    delete fjt_;
    fjt_ = nullptr;
}
void PotentialInt4C::compute_shell(
        const SGaussianShell &sh1,
        const SGaussianShell &sh2,
        const SGaussianShell &sh3,
        const SGaussianShell &sh4)
{
    bool p13p24_ = false, p12_ = false, p34_ = false;

    int oam1 = sh1.am();
    int oam2 = sh2.am();
    int oam3 = sh3.am();
    int oam4 = sh4.am();

    int n1 = sh1.ncartesian();
    int n2 = sh2.ncartesian();
    int n3 = sh3.ncartesian();
    int n4 = sh4.ncartesian();
    size_t size = n1 * n2 * n3 * n4;

    const SGaussianShell *s1, *s2, *s3, *s4, *temp;

    // l(a) >= l(b), l(c) >= l(d), and l(c) + l(d) >= l(a) + l(b).
    if (oam1 >= oam2) {
        s1 = &sh1;
        s2 = &sh2;
    } else {
        s1 = &sh2;
        s2 = &sh1;

        p12_ = true;
    }

    if (oam3 >= oam4) {
        s3 = &sh3;
        s4 = &sh4;
    } else {
        s3 = &sh4;
        s4 = &sh3;

        p34_ = true;
    }

    if ((oam1 + oam2) > (oam3 + oam4)) {
        // Swap s1 and s2 with s3 and s4
        temp = s1;
        s1 = s3;
        s3 = temp;

        temp = s2;
        s2 = s4;
        s4 = temp;

        p13p24_ = true;
    }

    // s1, s2, s3, s4 contain the shells to do in libint order

    int am1 = s1->am();
    int am2 = s2->am();
    int am3 = s3->am();
    int am4 = s4->am();
    int am = am1 + am2 + am3 + am4; // total am
    int nprim1;
    int nprim2;
    int nprim3;
    int nprim4;
    double A[3], B[3], C[3], D[3];

    A[0] = s1->x();
    A[1] = s1->y();
    A[2] = s1->z();
    B[0] = s2->x();
    B[1] = s2->y();
    B[2] = s2->z();
    C[0] = s3->x();
    C[1] = s3->y();
    C[2] = s3->z();
    D[0] = s4->x();
    D[1] = s4->y();
    D[2] = s4->z();

    // compute intermediates
    double AB2 = 0.0;
    AB2 += (A[0] - B[0]) * (A[0] - B[0]);
    AB2 += (A[1] - B[1]) * (A[1] - B[1]);
    AB2 += (A[2] - B[2]) * (A[2] - B[2]);
    double CD2 = 0.0;
    CD2 += (C[0] - D[0]) * (C[0] - D[0]);
    CD2 += (C[1] - D[1]) * (C[1] - D[1]);
    CD2 += (C[2] - D[2]) * (C[2] - D[2]);

    libint_.AB[0] = A[0] - B[0];
    libint_.AB[1] = A[1] - B[1];
    libint_.AB[2] = A[2] - B[2];
    libint_.CD[0] = C[0] - D[0];
    libint_.CD[1] = C[1] - D[1];
    libint_.CD[2] = C[2] - D[2];

    // Prepare all the data needed by libint
    size_t nprim = 0;
    nprim1 = s1->nprimitive();
    nprim2 = s2->nprimitive();
    nprim3 = s3->nprimitive();
    nprim4 = s4->nprimitive();

    const std::vector<double>&a1s = s1->es();
    const std::vector<double>&a2s = s2->es();
    const std::vector<double>&a3s = s3->es();
    const std::vector<double>&a4s = s4->es();
    const std::vector<double>&c1s = s1->cs();
    const std::vector<double>&c2s = s2->cs();
    const std::vector<double>&c3s = s3->cs();
    const std::vector<double>&c4s = s4->cs();

    // Old version - without ShellPair - STILL USED BY RI CODES
    for (int p1 = 0; p1 < nprim1; ++p1) {
        double a1 = a1s[p1];
        double c1 = c1s[p1];
        for (int p2 = 0; p2 < nprim2; ++p2) {
            double a2 = a2s[p2];
            double c2 = c2s[p2];
            double zeta = a1 + a2;
            double ooz = 1.0 / zeta;
            double oo2z = 1.0 / (2.0 * zeta);

            double PA[3], PB[3];
            double P[3];

            P[0] = (a1 * A[0] + a2 * B[0]) * ooz;
            P[1] = (a1 * A[1] + a2 * B[1]) * ooz;
            P[2] = (a1 * A[2] + a2 * B[2]) * ooz;
            PA[0] = P[0] - A[0];
            PA[1] = P[1] - A[1];
            PA[2] = P[2] - A[2];
            PB[0] = P[0] - B[0];
            PB[1] = P[1] - B[1];
            PB[2] = P[2] - B[2];

            double Sab = pow(M_PI * ooz, 3.0 / 2.0) * exp(-a1 * a2 * ooz * AB2) * c1 * c2;

            for (int p3 = 0; p3 < nprim3; ++p3) {
                double a3 = a3s[p3];
                double c3 = c3s[p3];
                for (int p4 = 0; p4 < nprim4; ++p4) {
                    double a4 = a4s[p4];
                    double c4 = c4s[p4];
                    double nu = a3 + a4;
                    double oon = 1.0 / nu;
                    double oo2n = 1.0 / (2.0 * nu);
                    double oo2zn = 1.0 / (2.0 * (zeta + nu));
                    double rho = (zeta * nu) / (zeta + nu);
                    double oo2rho = 1.0 / (2.0 * rho);

                    double QC[3], QD[3], WP[3], WQ[3], PQ[3];
                    double Q[3], W[3], a3C[3], a4D[3];

                    a3C[0] = a3 * C[0];
                    a3C[1] = a3 * C[1];
                    a3C[2] = a3 * C[2];

                    a4D[0] = a4 * D[0];
                    a4D[1] = a4 * D[1];
                    a4D[2] = a4 * D[2];

                    Q[0] = (a3C[0] + a4D[0]) * oon;
                    Q[1] = (a3C[1] + a4D[1]) * oon;
                    Q[2] = (a3C[2] + a4D[2]) * oon;

                    QC[0] = Q[0] - C[0];
                    QC[1] = Q[1] - C[1];
                    QC[2] = Q[2] - C[2];
                    QD[0] = Q[0] - D[0];
                    QD[1] = Q[1] - D[1];
                    QD[2] = Q[2] - D[2];
                    PQ[0] = P[0] - Q[0];
                    PQ[1] = P[1] - Q[1];
                    PQ[2] = P[2] - Q[2];

                    double PQ2 = 0.0;
                    PQ2 += (P[0] - Q[0]) * (P[0] - Q[0]);
                    PQ2 += (P[1] - Q[1]) * (P[1] - Q[1]);
                    PQ2 += (P[2] - Q[2]) * (P[2] - Q[2]);

                    W[0] = (zeta * P[0] + nu * Q[0]) / (zeta + nu);
                    W[1] = (zeta * P[1] + nu * Q[1]) / (zeta + nu);
                    W[2] = (zeta * P[2] + nu * Q[2]) / (zeta + nu);
                    WP[0] = W[0] - P[0];
                    WP[1] = W[1] - P[1];
                    WP[2] = W[2] - P[2];
                    WQ[0] = W[0] - Q[0];
                    WQ[1] = W[1] - Q[1];
                    WQ[2] = W[2] - Q[2];

                    for (int i = 0; i < 3; ++i) {
                        libint_.PrimQuartet[nprim].U[0][i] = PA[i];
                        libint_.PrimQuartet[nprim].U[2][i] = QC[i];
                        libint_.PrimQuartet[nprim].U[4][i] = WP[i];
                        libint_.PrimQuartet[nprim].U[5][i] = WQ[i];
                    }
                    libint_.PrimQuartet[nprim].oo2z = oo2z;
                    libint_.PrimQuartet[nprim].oo2n = oo2n;
                    libint_.PrimQuartet[nprim].oo2zn = oo2zn;
                    libint_.PrimQuartet[nprim].poz = rho * ooz;
                    libint_.PrimQuartet[nprim].pon = rho * oon;
                    libint_.PrimQuartet[nprim].oo2p = oo2rho;

                    double T = rho * PQ2;
                    fjt_->set_rho(rho);
                    double * F = fjt_->values(am, T);

                    // Modify F to include overlap of ab and cd, eqs 14, 15, 16 of libint manual
                    double Scd = pow(M_PI * oon, 3.0 / 2.0) * exp(-a3 * a4 * oon * CD2) * c3 * c4;
                    double val = 2.0 * sqrt(rho * M_1_PI) * Sab * Scd;
                    for (int i = 0; i <= am; ++i) {
                        libint_.PrimQuartet[nprim].F[i] = F[i] * val;
                    }
                    nprim++;
                }
            }
        }
    }

    // How many are there?
    // Compute the integral
    if (am) {
        double *target_ints;

        target_ints = build_eri[am1][am2][am3][am4](&libint_, nprim);

        memcpy(buffer2_, target_ints, sizeof(double) * size);
    }
    else {
        // Handle (ss|ss)
        double temp = 0.0;
        for (size_t i = 0; i < nprim; ++i)
            temp += (double) libint_.PrimQuartet[i].F[0];
        buffer2_[0] = temp;
    }

    // Transform the integrals into pure angular momentum
//  pure_transform(sh1, sh2, sh3, sh4, 1);

    // Permute integrals back, if needed
    if (p12_ || p34_ || p13p24_) {
        permute_target(buffer2_, buffer1_, s1, s2, s3, s4, p12_, p34_, p13p24_);
    }
    else {
        // copy the integrals to the target_
        memcpy(buffer1_, buffer2_, size * sizeof(double));
    }
}

void PotentialInt4C::compute_shell1(
        const SGaussianShell &sh1,
        const SGaussianShell &sh2,
        const SGaussianShell &sh3,
        const SGaussianShell &sh4)
{
    throw std::runtime_error("Not Implemented");
}
void PotentialInt4C::compute_shell2(
        const SGaussianShell &sh1,
        const SGaussianShell &sh2,
        const SGaussianShell &sh3,
        const SGaussianShell &sh4)
{
    throw std::runtime_error("Not Implemented");
}


} // namespace libgaussian
