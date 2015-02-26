#include <math.h>
#include <string.h>
#include <core/basisset.h>
#include <mints/schwarz.h>
#include <df/df.h>
#include "jk.h"

#include <omp.h>

using namespace ambit;

namespace lightspeed {

DFJK::DFJK(
    const std::shared_ptr<SchwarzSieve>& sieve,
    const std::shared_ptr<SBasisSet>& auxiliary) :
    JK(sieve),
    auxiliary_(auxiliary)
{
}
void DFJK::initialize()
{
    AODFERI df(sieve_,auxiliary_);
    df.set_doubles(doubles());
    df.set_a(a());
    df.set_b(b());
    df.set_w(w());
    df.set_metric_condition(metric_condition());

    if (!force_disk() && doubles() >= df.ao_task_core_doubles()) {
        is_core_ = true; 
        df_tensor_ = df.compute_ao_task_core(); 
    } else {
        is_core_ = false;
        df_tensor_ = df.compute_ao_task_disk(); 
    }
    initialized_ = true;
}
void DFJK::print(FILE* fh) const 
{
    fprintf(fh, "  DFJK:\n");
    fprintf(fh, "    Primary Basis    = %14s\n", primary_->name().c_str());
    fprintf(fh, "    Auxiliary Basis  = %14s\n", auxiliary_->name().c_str());
    fprintf(fh, "    Doubles          = %14zu\n", doubles_);
    fprintf(fh, "    Force Disk?      = %14zu\n", force_disk_ ? "Yes" : "No");
    fprintf(fh, "    Algorithm        = %14s\n", is_core_ ? "Core" : "Disk");
    fprintf(fh, "    Compute J?       = %14s\n", compute_J_ ? "Yes" : "No");
    fprintf(fh, "    Compute K?       = %14s\n", compute_K_ ? "Yes" : "No");
    fprintf(fh, "    Operator a       = %14.6E\n", a_);
    fprintf(fh, "    Operator b       = %14.6E\n", b_);
    fprintf(fh, "    Operator w       = %14.6E\n", w_);
    fprintf(fh, "    Product Cutoff   = %14.3E\n", product_cutoff_);
    fprintf(fh, "    Integral Cutoff  = %14.3E\n", sieve_->cutoff());
    fprintf(fh, "    Metric Condition = %14.3E\n", metric_condition_);
    fprintf(fh, "\n");
}
void DFJK::compute_JK_from_C(
    const std::vector<Tensor>& Ls,
    const std::vector<Tensor>& Rs,
    std::vector<Tensor>& Js,
    std::vector<Tensor>& Ks,
    const std::vector<double>& scaleJ,
    const std::vector<double>& scaleK)
{
    // => Check Inititalization <= //
    
    if (!initialized_) throw std::runtime_error("DFJK: Call initalize before computing J/K matrices");

    // => Scale Defaults <= //

    std::vector<double> sJ = (scaleJ.size() ? scaleJ : std::vector<double>(Ls.size(),1.0));
    std::vector<double> sK = (scaleK.size() ? scaleK : std::vector<double>(Ls.size(),1.0));

    // => Tasking <= //
    
    bool JK_symm = true;
    for (size_t ind = 0; ind < Ls.size(); ind++) {
        JK_symm = JK_symm && (Ls[ind] == Rs[ind]);
    }
    bool do_J = compute_J_;
    bool do_K = compute_K_;
    if (!do_J && !do_K) return; 

    // => Sizing <= //
    
    size_t natom   = primary_->natom();
    size_t nbf     = primary_->nfunction();
    size_t nthread = omp_get_max_threads();
}
void DFJK::compute_JK_from_D(
    const std::vector<Tensor>& Ds,
    const std::vector<bool>& symm,
    std::vector<Tensor>& Js,
    std::vector<Tensor>& Ks,
    const std::vector<double>& scaleJ,
    const std::vector<double>& scaleK)
{ 
    throw std::runtime_error("DFJK: Cannot do this yet.");
}
void DFJK::finalize()
{
    // TODO
}

} // namespace libgaussian
