#include <math.h>
#include <core/basisset.h>
#include <mints/schwarz.h>
#include "jk.h"

using namespace ambit;

namespace lightspeed {

JK::JK(const std::shared_ptr<SchwarzSieve>& sieve) :
    sieve_(sieve),
    primary_(sieve->basis1())
{
    if (!sieve_->is_symmetric()) throw std::runtime_error("JK sieve must be symmetric.");
}
void JK::initialize()
{
    throw std::runtime_error("Call a Derived class.");
}
void JK::print(
    FILE* fh) const
{
    throw std::runtime_error("Call a Derived class.");
}
void JK::compute_JK_from_C(
    const std::vector<Tensor>& L,
    const std::vector<Tensor>& R,
    std::vector<Tensor>& J,
    std::vector<Tensor>& K,
    const std::vector<double>& scaleJ,
    const std::vector<double>& scaleK)
{
    throw std::runtime_error("Call a Derived class.");
}
void JK::compute_JK_from_D(
    const std::vector<Tensor>& D,
    const std::vector<bool>& symm,
    std::vector<Tensor>& J,
    std::vector<Tensor>& K,
    const std::vector<double>& scaleJ,
    const std::vector<double>& scaleK)
{
    throw std::runtime_error("Call a Derived class.");
}
void JK::finalize()
{
    throw std::runtime_error("Call a Derived class.");
}

} // namespace libgaussian
