#include <math.h>
#include <libmints/int4c.h>
#include <libmints/sieve.h>
#include "df.h"

namespace libgaussian {

DFERI::DFERI(
    const std::shared_ptr<SBasisSet>& primary,
    const std::shared_ptr<SBasisSet>& auxiliary,
    const std::shared_ptr<SchwarzSieve>& sieve) :
    primary_(primary),
    auxiliary_(auxiliary),
    sieve_(sieve)
{
    doubles_ = 256000000L; // 1 GB
    a_ = 1.0;
    b_ = 0.0;
    w_ = 0.0;
    metric_condition_ = 1.0E-12;
}
Tensor DFERI::metric() const
{
    // TODO
}
Tensor DFERI::metric_power(
    double power,
    double condition) const 
{
    // TODO
}

AODFERI::AODFERI(
    const std::shared_ptr<SBasisSet>& primary,
    const std::shared_ptr<SBasisSet>& auxiliary,
    const std::shared_ptr<SchwarzSieve>& sieve) :
    DFERI(primary,auxiliary,sieve)
{
}
Tensor AODFERI::compute_ao_task_core(double power) 
{
    // TODO
}
Tensor AODFERI::compute_ao_task_disk(double power) 
{
    // TODO
}

MODFERI::MODFERI(
    const std::shared_ptr<SBasisSet>& primary,
    const std::shared_ptr<SBasisSet>& auxiliary,
    const std::shared_ptr<SchwarzSieve>& sieve) :
    DFERI(primary,auxiliary,sieve)
{
}
void MODFERI::clear()
{
    keys_.clear();
    Cls_.clear();
    Crs_.clear();
    powers_.clear();
    stripings_.clear();
}
void MODFERI::add_mo_task(
    const std::string& key,
    const Tensor& Cl,
    const Tensor& Cr,
    double power,
    const std::string& striping)
{
    std::vector<std::string> valid = { "lrQ", "rlQ",  "Qlr", "Qrl" };
    bool found = false;
    for (auto x : valid) {
        if (x == striping) found = true;    
    }
    if (!found) throw std::runtime_error("MODFERI: Invalid striping " + striping);

    keys_.push_back(key);
    Cls_[key] = Cl;
    Crs_[key] = Cr;
    powers_[key] = power;
    stripings_[key] = striping;
}
std::map<std::string, Tensor> MODFERI::compute_mo_tasks()
{
    // TODO
}

} // namespace libgaussian
