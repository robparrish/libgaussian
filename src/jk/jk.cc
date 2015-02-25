#include <math.h>
#include <core/basisset.h>
#include <mints/schwarz.h>
#include "jk.h"

using namespace tensor;

namespace libgaussian {

JK::JK(const std::shared_ptr<SchwarzSieve>& sieve) :
    sieve_(sieve),
    primary_(sieve->basis1())
{
    if (!sieve_->is_symmetric()) throw std::runtime_error("JK sieve must be symmetric.");
}


} // namespace libgaussian
