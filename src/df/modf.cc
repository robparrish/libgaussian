#include <math.h>
#include <mints/int4c.h>
#include <mints/schwarz.h>
#include "df.h"

#include <omp.h>

using namespace tensor;

namespace libgaussian {

MODFERI::MODFERI(
    const std::shared_ptr<SchwarzSieve>& sieve,
    const std::shared_ptr<SBasisSet>& auxiliary) :
    DFERI(sieve,auxiliary)
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
std::map<std::string, Tensor> MODFERI::compute_mo_tasks_core()
{
    std::map<std::string, Tensor> disk = compute_mo_tasks_disk();

    size_t memory = 0L;
    for (auto key : keys_) {
        memory += disk[key].numel(); 
    }
    if (memory > doubles()) throw std::runtime_error("MODFERI out of memory (switch to disk algorithm)");
    
    std::map<std::string, Tensor> core;
    
    for (auto key : keys_) {
        core[key] = disk[key].clone(kCore);
    }

    return core;
}
std::map<std::string, Tensor> MODFERI::compute_mo_tasks_disk()
{
    // TODO
    throw std::runtime_error("Not implemented.");
}

} // namespace libgaussian
