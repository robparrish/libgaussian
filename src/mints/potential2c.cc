#include <math.h>
#include <string.h>
#include <core/molecule.h>
#include "int2c.h"

namespace lightspeed {

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
    recursion_ = nullptr;

    size_t size;
    if (deriv_ == 0) {

        size = 1L * chunk_size();
        recursion_ = new ObaraSaikaTwoCenterVIRecursion(basis1->max_am() + 1, basis2->max_am() + 1);

    } else if (deriv_ == 1) {

        size = 3L * chunk_size() * basis1->natom();
        recursion_ = new ObaraSaikaTwoCenterVIDerivRecursion(basis1->max_am()+2, basis2->max_am()+2);

    } else {
        throw std::runtime_error("PotentialInt2C: deriv too high");
    }
    data1_.resize(size);
    data2_.resize(size);
    buffer1_ = data1_.data();
    buffer2_ = data2_.data();
}
PotentialInt2C::~PotentialInt2C()
{
    if (recursion_ != nullptr) delete recursion_;
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
void PotentialInt2C::compute_pair(
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

    const int izm = 1;
    const int iym = am1 + 1;
    const int ixm = iym * iym;
    const int jzm = 1;
    const int jym = am2 + 1;
    const int jxm = jym * jym;

    // compute intermediates
    double AB2 = 0.0;
    AB2 += (Ax - Bx) * (Ax - Bx);
    AB2 += (Ay - By) * (Ay - By);
    AB2 += (Az - Bz) * (Az - Bz);

    memset(buffer1_, 0, chunk_size() * sizeof(double));

    double ***vi = recursion_->vi();
    size_t ncharge = xs_.size();

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

            // loop over the charges
            for (size_t atom=0; atom<ncharge; ++atom) {
                double PC[3];

                double Z = Zs_[atom];

                PC[0] = Px - xs_[atom];
                PC[1] = Py - ys_[atom];
                PC[2] = Pz - zs_[atom];

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

    bool s1 = sh1.is_spherical();
    bool s2 = sh2.is_spherical();
    if (is_spherical_) apply_spherical(am1, am2, s1, s2, buffer1_, buffer2_);
}
void PotentialInt2C::compute_pair1(
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
    const size_t center_i = sh1.atom_index() * 3 * size;
    const size_t center_j = sh2.atom_index() * 3 * size;

    const int izm1 = 1;
    const int iym1 = am1 + 1 + 1;  // extra 1 for derivative
    const int ixm1 = iym1 * iym1;
    const int jzm1 = 1;
    const int jym1 = am2 + 1 + 1;  // extra 1 for derivative
    const int jxm1 = jym1 * jym1;

    // compute intermediates
    double AB2 = 0.0;
    AB2 += (Ax - Bx) * (Ax - Bx);
    AB2 += (Ay - By) * (Ay - By);
    AB2 += (Az - Bz) * (Az - Bz);

    memset(buffer1_, 0, 3 * basis1_->natom() * size * sizeof(double));

    double ***vi = recursion_->vi();
    double ***vx = recursion_->vx();
    double ***vy = recursion_->vy();
    double ***vz = recursion_->vz();
    const size_t ncharge = xs_.size();


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

            // Loop over atoms of basis set 1 (only works if bs1_ and bs2_ are on the same
            // molecule)
            for (int atom=0; atom<ncharge; ++atom) {
                double PC[3];

                double Z = Zs_[atom];

                PC[0] = Px - xs_[atom];
                PC[1] = Py - ys_[atom];
                PC[2] = Pz - zs_[atom];

                // Do recursion
                recursion_->compute(PA, PB, PC, gamma, am1+1, am2+1);

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
                                int iind = l1 * ixm1 + m1 * iym1 + n1 * izm1;
                                int jind = l2 * jxm1 + m2 * jym1 + n2 * jzm1;

                                const double pfac = over_pf * Z;

                                // x
                                double temp = 2.0*a1*vi[iind+ixm1][jind][0];
                                if (l1)
                                    temp -= l1*vi[iind-ixm1][jind][0];
                                buffer1_[center_i+(0*size)+ao12] -= temp * pfac;
                                // printf("ix temp = %f ", temp);

                                temp = 2.0*a2*vi[iind][jind+jxm1][0];
                                if (l2)
                                    temp -= l2*vi[iind][jind-jxm1][0];
                                buffer1_[center_j+(0*size)+ao12] -= temp * pfac;
                                // printf("jx temp = %f ", temp);

                                buffer1_[3*size*atom+ao12] -= vx[iind][jind][0] * pfac;

                                // y
                                temp = 2.0*a1*vi[iind+iym1][jind][0];
                                if (m1)
                                    temp -= m1*vi[iind-iym1][jind][0];
                                buffer1_[center_i+(1*size)+ao12] -= temp * pfac;
                                // printf("iy temp = %f ", temp);

                                temp = 2.0*a2*vi[iind][jind+jym1][0];
                                if (m2)
                                    temp -= m2*vi[iind][jind-jym1][0];
                                buffer1_[center_j+(1*size)+ao12] -= temp * pfac;
                                // printf("jy temp = %f ", temp);

                                buffer1_[3*size*atom+size+ao12] -= vy[iind][jind][0] * pfac;

                                // z
                                temp = 2.0*a1*vi[iind+izm1][jind][0];
                                if (n1)
                                    temp -= n1*vi[iind-izm1][jind][0];
                                buffer1_[center_i+(2*size)+ao12] -= temp * pfac;
                                // printf("iz temp = %f ", temp);

                                temp = 2.0*a2*vi[iind][jind+jzm1][0];
                                if (n2)
                                    temp -= n2*vi[iind][jind-jzm1][0];
                                buffer1_[center_j+(2*size)+ao12] -= temp * pfac;
                                // printf("jz temp = %f \n", temp);

                                buffer1_[3*size*atom+2*size+ao12] -= vz[iind][jind][0] * pfac;

                                ao12++;
                            }
                        }
                    }
                }
            }
        }
    }

    bool s1 = sh1.is_spherical();
    bool s2 = sh2.is_spherical();
    if (is_spherical_) {
        for (size_t atom=0; atom < basis1_->natom(); ++atom)
            apply_spherical(am1, am2, s1, s2, buffer1_ + atom * chunk_size(), buffer2_ + atom * chunk_size());
    }
}
void PotentialInt2C::compute_pair2(
    const SGaussianShell& sh1,
    const SGaussianShell& sh2)
{
    throw std::runtime_error("Not Implemented");
}


} // namespace lightspeed
