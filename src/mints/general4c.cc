#include <math.h>
#include <string.h>
#include "int4c.h"
#include "constants.h"
#include "fjt.h"

#include <libint2.h>
#include <libint2/libint2_types.h>

namespace lightspeed {

struct GeneralInt4C::Impl
{
    Impl(const std::shared_ptr<SBasisSet> &basis1,
         const std::shared_ptr<SBasisSet> &basis2,
         const std::shared_ptr<SBasisSet> &basis3,
         const std::shared_ptr<SBasisSet> &basis4,
         const std::shared_ptr<fundamentals::base::Fjt> &fjt) :
            fjt_shared_(fjt),
            fjt_(fjt.get()),
            libint_(basis1->max_nprimitive() * basis2->max_nprimitive() * basis3->max_nprimitive() * basis4->max_nprimitive())
    {
        libint2_static_init();
        libint2_init_eri(&libint_[0],
                         std::max(
                                 std::max(
                                         basis1->max_am(),
                                         basis2->max_am()),
                                 std::max(
                                         basis3->max_am(),
                                         basis4->max_am()
                                 )
                         ), 0);
    }
    virtual ~Impl()
    {
        libint2_cleanup_eri(&libint_[0]);
        libint2_static_cleanup();
    }
    std::shared_ptr<fundamentals::base::Fjt> fjt_shared_;
    fundamentals::base::Fjt *fjt_;
    std::vector<Libint_t> libint_;
};

GeneralInt4C::GeneralInt4C(
        const std::shared_ptr<SBasisSet> &basis1,
        const std::shared_ptr<SBasisSet> &basis2,
        const std::shared_ptr<SBasisSet> &basis3,
        const std::shared_ptr<SBasisSet> &basis4,
        const std::shared_ptr<fundamentals::base::Fjt> &fundamental,
        int deriv) :
        Int4C(basis1, basis2, basis3, basis4, deriv),
        impl_(new Impl(basis1, basis2, basis3, basis4, fundamental))
{
    size_t size;
    if (deriv_ == 0) {
        size = 1L * chunk_size();
    } else {
        throw std::runtime_error("GeneralInt4C: deriv too high");
    }
    data1_.resize(size);
    data2_.resize(size);
    buffer1_ = data1_.data();
    buffer2_ = data2_.data();
}

GeneralInt4C::~GeneralInt4C()
{
    delete impl_;
    impl_ = nullptr;
}

void GeneralInt4C::compute_quartet(
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

    size_t n1 = sh1.ncartesian();
    size_t n2 = sh2.ncartesian();
    size_t n3 = sh3.ncartesian();
    size_t n4 = sh4.ncartesian();
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

    const int am1 = s1->am();
    const int am2 = s2->am();
    const int am3 = s3->am();
    const int am4 = s4->am();
    const int am = am1 + am2 + am3 + am4; // total am
    const size_t nprim1 = s1->nprimitive();
    const size_t nprim2 = s2->nprimitive();
    const size_t nprim3 = s3->nprimitive();
    const size_t nprim4 = s4->nprimitive();
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

    // Prepare all the data needed by libint
    size_t nprim = 0;

    const std::vector<double> &a1s = s1->es();
    const std::vector<double> &a2s = s2->es();
    const std::vector<double> &a3s = s3->es();
    const std::vector<double> &a4s = s4->es();
    const std::vector<double> &c1s = s1->cs();
    const std::vector<double> &c2s = s2->cs();
    const std::vector<double> &c3s = s3->cs();
    const std::vector<double> &c4s = s4->cs();

    const std::vector<double> &F = impl_->fjt_->values();

    for (size_t p1 = 0; p1 < nprim1; ++p1) {
        double a1 = a1s[p1];
        double c1 = c1s[p1];
        for (size_t p2 = 0; p2 < nprim2; ++p2) {
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

            for (size_t p3 = 0; p3 < nprim3; ++p3) {
                double a3 = a3s[p3];
                double c3 = c3s[p3];
                for (size_t p4 = 0; p4 < nprim4; ++p4) {
                    double a4 = a4s[p4];
                    double c4 = c4s[p4];
                    double nu = a3 + a4;
                    double oon = 1.0 / nu;
                    double oo2n = 1.0 / (2.0 * nu);
                    double oo2zn = 1.0 / (2.0 * (zeta + nu));
                    double rho = (zeta * nu) / (zeta + nu);

                    double QD[3], PQ[3];
                    double Q[3], W[3], a3C[3], a4D[3];

                    Libint_t &eri = impl_->libint_[nprim];

                    eri.AB_x[0] = A[0] - B[0];
                    eri.AB_y[0] = A[1] - B[1];
                    eri.AB_z[0] = A[2] - B[2];
                    eri.CD_x[0] = C[0] - D[0];
                    eri.CD_y[0] = C[1] - D[1];
                    eri.CD_z[0] = C[2] - D[2];

                    a3C[0] = a3 * C[0];
                    a3C[1] = a3 * C[1];
                    a3C[2] = a3 * C[2];

                    a4D[0] = a4 * D[0];
                    a4D[1] = a4 * D[1];
                    a4D[2] = a4 * D[2];

                    Q[0] = (a3C[0] + a4D[0]) * oon;
                    Q[1] = (a3C[1] + a4D[1]) * oon;
                    Q[2] = (a3C[2] + a4D[2]) * oon;

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

                    eri.PA_x[0] = PA[0];
                    eri.PA_y[0] = PA[1];
                    eri.PA_z[0] = PA[2];

                    eri.QC_x[0] = Q[0] - C[0];
                    eri.QC_y[0] = Q[1] - C[1];
                    eri.QC_z[0] = Q[2] - C[2];

                    eri.WP_x[0] = W[0] - P[0];
                    eri.WP_y[0] = W[1] - P[1];
                    eri.WP_z[0] = W[2] - P[2];

                    eri.WQ_x[0] = W[0] - Q[0];
                    eri.WQ_y[0] = W[1] - Q[1];
                    eri.WQ_z[0] = W[2] - Q[2];

                    eri.oo2z[0] = oo2z;
                    eri.oo2e[0] = oo2n;
                    eri.oo2ze[0] = oo2zn;
                    eri.roz[0] = rho * ooz;
                    eri.roe[0] = rho * oon;

                    double T = rho * PQ2;
                    impl_->fjt_->set_rho(rho);
                    impl_->fjt_->compute(am, T);

                    // Modify F to include overlap of ab and cd, eqs 14, 15, 16 of libint manual
                    double Scd = pow(M_PI * oon, 3.0 / 2.0) * exp(-a3 * a4 * oon * CD2) * c3 * c4;
                    double val = 2.0 * sqrt(rho * M_1_PI) * Sab * Scd;
                    for (int i = 0; i <= am; ++i) {
                        eri.LIBINT_T_SS_EREP_SS(0)[i] = F[i] * val;
                    }
                    nprim++;
                }
            }
        }
    }

    impl_->libint_[0].contrdepth = nprim;

    // How many are there?
    // Compute the integral
    if (am) {
        libint2_build_eri[am1][am2][am3][am4](impl_->libint_.data());

        memcpy(buffer2_, impl_->libint_[0].targets[0], sizeof(double) * size);
    }
    else {
        // Handle (ss|ss)
        double temp = 0.0;
        for (size_t i = 0; i < nprim; ++i)
            temp += impl_->libint_[i].LIBINT_T_SS_EREP_SS(0)[0];
        buffer2_[0] = temp;
    }

    // Transform the integrals into pure angular momentum
    bool t1 = s1->is_spherical();
    bool t2 = s2->is_spherical();
    bool t3 = s3->is_spherical();
    bool t4 = s4->is_spherical();
    if (is_spherical_) apply_spherical(am1, am2, am3, am4, t1, t2, t3, t4, buffer2_, buffer1_);

    // Permute integrals back, if needed
    if (p12_ || p34_ || p13p24_) {
        permute_target(buffer2_, buffer1_, s1, s2, s3, s4, p12_, p34_, p13p24_);
    }
    else {
        // copy the integrals to the target_
        memcpy(buffer1_, buffer2_, size * sizeof(double));
    }
}

void GeneralInt4C::compute_quartet1(
        const SGaussianShell &sh1,
        const SGaussianShell &sh2,
        const SGaussianShell &sh3,
        const SGaussianShell &sh4)
{
    throw std::runtime_error("Not Implemented");
}
void GeneralInt4C::compute_quartet2(
        const SGaussianShell &sh1,
        const SGaussianShell &sh2,
        const SGaussianShell &sh3,
        const SGaussianShell &sh4)
{
    throw std::runtime_error("Not Implemented");
}

F12Int4C::F12Int4C(const std::shared_ptr<SBasisSet> &basis1, const std::shared_ptr<SBasisSet> &basis2,
                   const std::shared_ptr<SBasisSet> &basis3, const std::shared_ptr<SBasisSet> &basis4,
                   const fundamentals::parameters::CorrelationFactor &cf, int deriv)
        : GeneralInt4C(basis1,
                       basis2,
                       basis3,
                       basis4,
                       std::make_shared<fundamentals::F12>(cf,
                                                           basis1->max_am() +
                                                                   basis2->max_am() +
                                                                   basis3->max_am() +
                                                                   basis4->max_am() +
                                                                   deriv + 1),
                       deriv)
{
}

F12ScaledInt4C::F12ScaledInt4C(const std::shared_ptr<SBasisSet> &basis1, const std::shared_ptr<SBasisSet> &basis2,
                               const std::shared_ptr<SBasisSet> &basis3, const std::shared_ptr<SBasisSet> &basis4,
                               const fundamentals::parameters::CorrelationFactor &cf, int deriv)
        : GeneralInt4C(basis1,
                       basis2,
                       basis3,
                       basis4,
                       std::make_shared<fundamentals::F12Scaled>(cf,
                                                                 basis1->max_am() +
                                                                         basis2->max_am() +
                                                                         basis3->max_am() +
                                                                         basis4->max_am() +
                                                                         deriv + 1),
                       deriv)
{
}

F12SquaredInt4C::F12SquaredInt4C(const std::shared_ptr<SBasisSet> &basis1, const std::shared_ptr<SBasisSet> &basis2,
                                 const std::shared_ptr<SBasisSet> &basis3, const std::shared_ptr<SBasisSet> &basis4,
                                 const fundamentals::parameters::CorrelationFactor &cf, int deriv)
        : GeneralInt4C(basis1,
                       basis2,
                       basis3,
                       basis4,
                       std::make_shared<fundamentals::F12Squared>(cf,
                                                                  basis1->max_am() +
                                                                          basis2->max_am() +
                                                                          basis3->max_am() +
                                                                          basis4->max_am() +
                                                                          deriv + 1),
                       deriv)
{
}

F12G12Int4C::F12G12Int4C(const std::shared_ptr<SBasisSet> &basis1, const std::shared_ptr<SBasisSet> &basis2,
                         const std::shared_ptr<SBasisSet> &basis3, const std::shared_ptr<SBasisSet> &basis4,
                         const fundamentals::parameters::CorrelationFactor &cf, int deriv)
        : GeneralInt4C(basis1,
                       basis2,
                       basis3,
                       basis4,
                       std::make_shared<fundamentals::F12G12>(cf,
                                                              basis1->max_am() +
                                                                      basis2->max_am() +
                                                                      basis3->max_am() +
                                                                      basis4->max_am() +
                                                                      deriv + 1),
                       deriv)
{
}

F12DoubleCommutatorInt4C::F12DoubleCommutatorInt4C(const std::shared_ptr<SBasisSet> &basis1,
                                                   const std::shared_ptr<SBasisSet> &basis2,
                                                   const std::shared_ptr<SBasisSet> &basis3,
                                                   const std::shared_ptr<SBasisSet> &basis4,
                                                   const fundamentals::parameters::CorrelationFactor &cf, int deriv)
        : GeneralInt4C(basis1,
                       basis2,
                       basis3,
                       basis4,
                       std::make_shared<fundamentals::F12DoubleCommutator>(cf,
                                                                           basis1->max_am() +
                                                                                   basis2->max_am() +
                                                                                   basis3->max_am() +
                                                                                   basis4->max_am() +
                                                                                   deriv + 1),
                       deriv)
{
}


}
