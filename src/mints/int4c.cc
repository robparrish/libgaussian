#include "int4c.h"

namespace libgaussian {

Int4C::Int4C(
    const std::shared_ptr<SBasisSet>& basis1,
    const std::shared_ptr<SBasisSet>& basis2,
    const std::shared_ptr<SBasisSet>& basis3,
    const std::shared_ptr<SBasisSet>& basis4,
    int deriv) :
    basis1_(basis1),
    basis2_(basis2),
    basis3_(basis3),
    basis4_(basis4),
    deriv_(deriv)
{
    buffer1_ = nullptr;
    buffer2_ = nullptr;
    am_info_ = SAngularMomentum::build(max_am());
}
Int4C::Int4C()
{
    buffer1_ = nullptr;
    buffer2_ = nullptr;
}
Int4C::~Int4C()
{
    if (buffer1_ != nullptr) delete[] buffer1_;
    if (buffer2_ != nullptr) delete[] buffer2_;
}
int Int4C::max_am() const 
{
    return std::max(
            std::max(
            basis1_->max_am(),
            basis2_->max_am()),
            std::max(
            basis3_->max_am(),
            basis4_->max_am()));
}
int Int4C::total_am() const 
{
    return 
        basis1_->max_am() + 
        basis2_->max_am() +
        basis3_->max_am() +
        basis4_->max_am();
}
size_t Int4C::chunk_size() const 
{
    return 
        basis1_->max_ncartesian() * 
        basis2_->max_ncartesian() * 
        basis3_->max_ncartesian() * 
        basis4_->max_ncartesian();
}
void Int4C::compute_shell(
    size_t shell1,
    size_t shell2,
    size_t shell3,
    size_t shell4)
{
    compute_shell(
        basis1_->shell(shell1),
        basis2_->shell(shell2),
        basis3_->shell(shell3),
        basis4_->shell(shell4));
}
void Int4C::compute_shell1(
    size_t shell1,
    size_t shell2,
    size_t shell3,
    size_t shell4)
{
    compute_shell1(
        basis1_->shell(shell1),
        basis2_->shell(shell2),
        basis3_->shell(shell3),
        basis4_->shell(shell4));
}
void Int4C::compute_shell2(
    size_t shell1,
    size_t shell2,
    size_t shell3,
    size_t shell4)
{
    compute_shell2(
        basis1_->shell(shell1),
        basis2_->shell(shell2),
        basis3_->shell(shell3),
        basis4_->shell(shell4));
}
void Int4C::compute_shell(
    const SGaussianShell& sh1,
    const SGaussianShell& sh2,
    const SGaussianShell& sh3,
    const SGaussianShell& sh4)
{
    throw std::runtime_error("Int4C: compute_shell not implemented for this type");
}
void Int4C::compute_shell1(
    const SGaussianShell& sh1,
    const SGaussianShell& sh2,
    const SGaussianShell& sh3,
    const SGaussianShell& sh4)
{
    throw std::runtime_error("Int4C: compute_shell1 not implemented for this type");
}
void Int4C::compute_shell2(
    const SGaussianShell& sh1,
    const SGaussianShell& sh2,
    const SGaussianShell& sh3,
    const SGaussianShell& sh4)
{
    throw std::runtime_error("Int4C: compute_shell2 not implemented for this type");
}

} // namespace libgaussian
