#include <math.h>
#include <core/molecule.h>
#include "int2c.h"

namespace libgaussian {

PotentialInt2C::PotentialInt2C(
    const std::shared_ptr<SBasisSet>& basis1,
    const std::shared_ptr<SBasisSet>& basis2,
    int deriv,
    double a,
    double b,
    double w) :
    Int2C(basis1,basis2,deriv),
    a_(a),
    b_(b),
    w_(w)
{
    size_t size;
    if (deriv_ == 0) {
        size = 1L * chunk_size();

        recursion_ =  new ObaraSaikaTwoCenterVIRecursion(basis1->max_am()+1, basis2->max_am()+1);
    } else {
        throw std::runtime_error("PotentialInt2C: deriv too high");
    }
    buffer1_ = new double[size];
    buffer2_ = new double[size];
}
void PotentialInt2C::set_nuclear_potential(
    const std::shared_ptr<SMolecule>& mol,
    bool use_nuclear)
{
    xs_.resize(mol->natom());
    ys_.resize(mol->natom());
    zs_.resize(mol->natom());
    Zs_.resize(mol->natom());

    for (size_t ind = 0; ind < mol->natom(); ind++) {
        xs_[ind] = mol->atom(ind).x();
        ys_[ind] = mol->atom(ind).y();
        zs_[ind] = mol->atom(ind).z();
        Zs_[ind] = (use_nuclear ?
            mol->atom(ind).Z() :
            mol->atom(ind).Q());
    }
}
void PotentialInt2C::compute_shell(
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

    int izm = 1;
    int iym = am1 + 1;
    int ixm = iym * iym;
    int jzm = 1;
    int jym = am2 + 1;
    int jxm = jym * jym;

    // compute intermediates
    double AB2 = 0.0;
    AB2 += (A[0] - B[0]) * (A[0] - B[0]);
    AB2 += (A[1] - B[1]) * (A[1] - B[1]);
    AB2 += (A[2] - B[2]) * (A[2] - B[2]);

    memset(buffer1_, 0, sh1.ncartesian() * sh2.ncartesian() * sizeof(double));

    double ***vi = recursion_->vi();
    int ncharge = xs_.size();

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

            // loop over the charges
            for (int atom=0; atom<ncharge; ++atom) {
                double PC[3];

                double Z = Zs_[atom];

                PC[0] = P[0] - xs_[atom];
                PC[1] = P[1] - ys_[atom];
                PC[2] = P[2] - zs_[atom];

                // Do recursion
                recursion_->compute(PA, PB, PC, gamma, am1, am2);

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

                                // Compute location in the recursion
                                int iind = l1 * ixm + m1 * iym + n1 * izm;
                                int jind = l2 * jxm + m2 * jym + n2 * jzm;

                                buffer1_[ao12++] += -vi[iind][jind][0] * over_pf * Z;
                            }
                        }
                    }
                }
            }
        }
    }
}
void PotentialInt2C::compute_shell1(
    const SGaussianShell& sh1,
    const SGaussianShell& sh2)
{
    throw std::runtime_error("Not Implemented");
}
void PotentialInt2C::compute_shell2(
    const SGaussianShell& sh1,
    const SGaussianShell& sh2)
{
    throw std::runtime_error("Not Implemented");
}


} // namespace libgaussian
