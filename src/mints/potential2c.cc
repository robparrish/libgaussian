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
    throw std::runtime_error("Not Implemented");
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
