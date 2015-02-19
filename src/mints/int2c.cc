#include "int2c.h"

namespace libgaussian {

Int2C::Int2C(
    const std::shared_ptr<SBasisSet>& basis1,
    const std::shared_ptr<SBasisSet>& basis2,
    int deriv) :
    basis1_(basis1),
    basis2_(basis2),
    deriv_(deriv)
{
    buffer1_ = nullptr;
    buffer2_ = nullptr;
    x_ = 0.0;
    y_ = 0.0;
    z_ = 0.0;
    am_info_ = SAngularMomentum::build(max_am());
}
Int2C::Int2C()
{
    buffer1_ = nullptr;
    buffer2_ = nullptr;
}
Int2C::~Int2C()
{
    if (buffer1_ != nullptr) delete[] buffer1_;
    if (buffer2_ != nullptr) delete[] buffer2_;
}
int Int2C::max_am() const 
{
    return std::max(basis1_->max_am(),basis2_->max_am());
}
int Int2C::total_am() const 
{
    return basis1_->max_am() + basis2_->max_am();
}
size_t Int2C::chunk_size() const 
{
    return basis1_->max_ncartesian() * basis2_->max_ncartesian();
}
void Int2C::compute_shell(
    size_t shell1,
    size_t shell2)
{
    compute_shell(basis1_->shell(shell1),basis2_->shell(shell2));
}
void Int2C::compute_shell1(
    size_t shell1,
    size_t shell2)
{
    compute_shell1(basis1_->shell(shell1),basis2_->shell(shell2));
}
void Int2C::compute_shell2(
    size_t shell1,
    size_t shell2)
{
    compute_shell2(basis1_->shell(shell1),basis2_->shell(shell2));
}
void Int2C::compute_shell(
    const SGaussianShell& sh1,
    const SGaussianShell& sh2)
{
    throw std::runtime_error("Int2C: compute_shell not implemented for this type");
}
void Int2C::compute_shell1(
    const SGaussianShell& sh1,
    const SGaussianShell& sh2)
{
    throw std::runtime_error("Int2C: compute_shell1 not implemented for this type");
}
void Int2C::compute_shell2(
    const SGaussianShell& sh1,
    const SGaussianShell& sh2)
{
    throw std::runtime_error("Int2C: compute_shell2 not implemented for this type");
}

} // namespace libgaussian
