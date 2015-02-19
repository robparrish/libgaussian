#include <math.h>
#include "int2c.h"

namespace libgaussian {

KineticInt2C::KineticInt2C(
    const std::shared_ptr<SBasisSet>& basis1,
    const std::shared_ptr<SBasisSet>& basis2,
    int deriv) :
    Int2C(basis1,basis2,deriv)
{
    size_t size;
    if (deriv_ == 0) {
        size = 1L * chunk_size();
    } else {
        throw std::runtime_error("KineticInt2C: deriv too high");
    }
    buffer1_ = new double[size];
    buffer2_ = new double[size];
}
void KineticInt2C::compute_shell(
    const SGaussianShell& sh1,
    const SGaussianShell& sh2)
{
    throw std::runtime_error("Not Implemented");
}
void KineticInt2C::compute_shell1(
    const SGaussianShell& sh1,
    const SGaussianShell& sh2)
{
    throw std::runtime_error("Not Implemented");
}
void KineticInt2C::compute_shell2(
    const SGaussianShell& sh1,
    const SGaussianShell& sh2)
{
    throw std::runtime_error("Not Implemented");
}


} // namespace libgaussian
