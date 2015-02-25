#include <math.h>
#include "int2c.h"

namespace lightspeed {

QuadrupoleInt2C::QuadrupoleInt2C(
    const std::shared_ptr<SBasisSet>& basis1,
    const std::shared_ptr<SBasisSet>& basis2,
    int deriv) :
    Int2C(basis1,basis2,deriv)
{
    size_t size;
    if (deriv_ == 0) {
        size = 6L * chunk_size();
    } else {
        throw std::runtime_error("QuadrupoleInt2C: deriv too high");
    }
    data1_.resize(size);
    data2_.resize(size);
    buffer1_ = data1_.data();
    buffer2_ = data2_.data();
}
void QuadrupoleInt2C::compute_pair(
    const SGaussianShell& sh1,
    const SGaussianShell& sh2)
{
    throw std::runtime_error("Not Implemented");
}
void QuadrupoleInt2C::compute_pair1(
    const SGaussianShell& sh1,
    const SGaussianShell& sh2)
{
    throw std::runtime_error("Not Implemented");
}
void QuadrupoleInt2C::compute_pair2(
    const SGaussianShell& sh1,
    const SGaussianShell& sh2)
{
    throw std::runtime_error("Not Implemented");
}


} // namespace lightspeed
