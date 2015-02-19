#include <math.h>
#include "int4c.h"

namespace libgaussian {

PotentialInt4C::PotentialInt4C(
    const std::shared_ptr<SBasisSet>& basis1,
    const std::shared_ptr<SBasisSet>& basis2,
    const std::shared_ptr<SBasisSet>& basis3,
    const std::shared_ptr<SBasisSet>& basis4,
    int deriv,
    double a,
    double b,
    double w) :
    Int4C(basis1,basis2,basis3,basis4,deriv),
    a_(a),
    b_(b),
    w_(w)
{
    size_t size;
    if (deriv_ == 0) {
        size = 1L * chunk_size();
    } else {
        throw std::runtime_error("PotentialInt4C: deriv too high");
    }
    buffer1_ = new double[size];
    buffer2_ = new double[size];
}
void PotentialInt4C::compute_shell(
    const SGaussianShell& sh1,
    const SGaussianShell& sh2,
    const SGaussianShell& sh3,
    const SGaussianShell& sh4)
{
    throw std::runtime_error("Not Implemented");
}
void PotentialInt4C::compute_shell1(
    const SGaussianShell& sh1,
    const SGaussianShell& sh2,
    const SGaussianShell& sh3,
    const SGaussianShell& sh4)
{
    throw std::runtime_error("Not Implemented");
}
void PotentialInt4C::compute_shell2(
    const SGaussianShell& sh1,
    const SGaussianShell& sh2,
    const SGaussianShell& sh3,
    const SGaussianShell& sh4)
{
    throw std::runtime_error("Not Implemented");
}


} // namespace libgaussian
