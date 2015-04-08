#include "tb.h"
#include <mints/schwarz.h>
#include <core/basisset.h>

namespace lightspeed {

TwoBody::TwoBody(const std::shared_ptr<SchwarzSieve>& sieve12,
        const std::shared_ptr<SchwarzSieve>& sieve34) :
        sieve12_(sieve12),
        sieve34_(sieve34),
        basis1_(sieve12_->basis1()),
        basis2_(sieve12_->basis2()),
        basis3_(sieve34_->basis1()),
        basis4_(sieve34_->basis2())
{
}

MOTwoBody::MOTwoBody(const std::shared_ptr<SchwarzSieve>& sieve12,
                 const std::shared_ptr<SchwarzSieve>& sieve34) :
        TwoBody(sieve12,sieve34)
{
}

void MOTwoBody::clear()
{
    keys_.clear();
    C1s_.clear();
    C2s_.clear();
    C3s_.clear();
    C4s_.clear();
    stripings_.clear();
}
void MOTwoBody::add_mo_task(
        const std::string &key,
        const ambit::Tensor &C1,
        const ambit::Tensor &C2,
        const ambit::Tensor &C3,
        const ambit::Tensor &C4,
        std::string const &striping)
{
    if (C1s_.count(key)) throw std::runtime_error("MODFERI: Duplicate key " + key);

    if (C1.rank() != 2) throw std::runtime_error("MODFERI:: C1 must be rank 2");
    if (C1.dim(0) != basis1_->nfunction()) throw std::runtime_error("MODFERI: C1 must be nbf x norb");

    if (C2.rank() != 2) throw std::runtime_error("MODFERI:: C2 must be rank 2");
    if (C2.dim(0) != basis2_->nfunction()) throw std::runtime_error("MODFERI: C2 must be nbf x norb");

    if (C3.rank() != 2) throw std::runtime_error("MODFERI:: C3 must be rank 2");
    if (C3.dim(0) != basis3_->nfunction()) throw std::runtime_error("MODFERI: C3 must be nbf x norb");

    if (C4.rank() != 2) throw std::runtime_error("MODFERI:: C4 must be rank 2");
    if (C4.dim(0) != basis4_->nfunction()) throw std::runtime_error("MODFERI: C4 must be nbf x norb");

    // TODO Check Striping

    keys_.push_back(key);
    C1s_[key] = C1;
    C2s_[key] = C2;
    C3s_[key] = C3;
    C4s_[key] = C4;
    stripings_[key] = striping;
}
std::map<std::string, ambit::Tensor> MOTwoBody::compute_mo_tasks_core() const
{
    // TODO
}
std::map<std::string, ambit::Tensor> MOTwoBody::compute_mo_tasks_disk() const
{
    // TODO
}



}
