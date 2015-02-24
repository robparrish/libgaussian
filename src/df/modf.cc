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
    if (Cls_.count(key)) throw std::runtime_error("MODFERI: Duplicate key " + key);

    if (Cl.rank() != 2) throw std::runtime_error("MODFERI:: Cl must be rank 2");
    if (Cl.dim(0) != primary_->nfunction()) throw std::runtime_error("MODFERI: Cl must be nbf x norb");

    if (Cr.rank() != 2) throw std::runtime_error("MODFERI:: Cr must be rank 2");
    if (Cr.dim(0) != primary_->nfunction()) throw std::runtime_error("MODFERI: Cr must be nbf x norb");

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
std::map<std::string, Tensor> MODFERI::compute_mo_tasks_core() const
{
    size_t memory = 0L;
    for (auto key : keys_) {
        memory += auxiliary_->nfunction() * 
            Cls_.at(key).dim(1) *
            Crs_.at(key).dim(1);
    }
    if (memory > doubles()) throw std::runtime_error("MODFERI out of memory (switch to disk algorithm)");

    std::map<std::string, Tensor> disk = compute_mo_tasks_disk();

    std::map<std::string, Tensor> core;
    for (auto key : keys_) {
        core[key] = disk[key].clone(kCore);
    }

    return core;
}
std::map<std::string, Tensor> MODFERI::compute_mo_tasks_disk() const
{
    if (3L * auxiliary_->nfunction() * auxiliary_->nfunction() > doubles()) 
        throw std::runtime_error("MODFERI: out of memory");

    std::map<std::string, Tensor> Aia  = transform();
    std::map<std::string, Tensor> iaQ  = fit(Aia);
    return iaQ;
}
std::map<std::string, tensor::Tensor> MODFERI::transform() const
{
    // => Sizing <= //

    size_t nbf  = primary_->nfunction(); 
    size_t naux = auxiliary_->nfunction(); 

    int nthread = omp_get_max_threads();

    // => First-Half Transform Merging <= //

    std::vector<std::vector<std::string>> tasks;
    for (size_t kind = 0; kind < keys_.size(); kind++) {
        std::string key = keys_[kind];
        bool found = true;
        for (size_t tind = 0; tind < tasks.size(); tind++) {
            if (Cls_.at(key) == Cls_.at(tasks[tind][0])) {
                tasks[tind].push_back(key);
                found = true;
                break;
            }    
        }
        if (!found) tasks.push_back({key});
    }

    // => Maximum Orbital Sizes <= //

    size_t maxpq = nbf * nbf;
    size_t max1q = 0L;
    size_t max12 = 0L;  
    for (auto key : keys_) {
        max1q = std::max(Cls_.at(key).dim(1)*nbf,max1q);
        max12 = std::max(Cls_.at(key).dim(1)*Crs_.at(key).dim(1),max12);
    }

    // => Memory Blocking (Auxiliary Blocks) <= //

    size_t per_row = maxpq + max1q + max12;
    size_t max_rows = (doubles() / per_row);
    max_rows = std::min(max_rows,naux);
    if (max_rows < auxiliary_->max_nfunction()) throw std::runtime_error("MODFERI: Out of memory.");

    // => Shell Block Assignments <= //

    std::vector<size_t> Ashell_tasks;
    Ashell_tasks.push_back(0);
    for (size_t A = 0; A < auxiliary_->nshell(); A++) {
        size_t start = auxiliary_->shell(Ashell_tasks.back()).function_index();
        size_t stop = auxiliary_->shell(A).function_index();
        if (stop - start > max_rows) {
            Ashell_tasks.push_back(A);
        }
    }
    Ashell_tasks.push_back(auxiliary_->nshell());

    // => Temporaries <= //

    Tensor Apq = Tensor::build(kCore, "Apq", {max_rows, maxpq});
    Tensor Api = Tensor::build(kCore, "Api", {max_rows, max1q});
    Tensor Aia = Tensor::build(kCore, "Aia", {max_rows, max12});
    double* Ap = Apq.data().data();
    
    // => Targets <= //

    std::map<std::string, Tensor> targets;
    for (auto key : keys_) {
        const std::string& stripe = stripings_.at(key);
        std::vector<size_t> dims;
        if (stripe == "lrQ" || stripe == "Qlr") {
            dims = {naux, Cls_.at(key).dim(1)*Crs_.at(key).dim(1)};
        } else {
            dims = {naux, Crs_.at(key).dim(1)*Cls_.at(key).dim(1)};
        }
        targets[key] = Tensor::build(kDisk, key + "_ints", dims); 
    }

    // => Integrals <= //

    std::shared_ptr<SBasisSet> zero = SBasisSet::zero_basis();
    std::vector<std::shared_ptr<PotentialInt4C>> ints;
    for (int t = 0; t < nthread; t++) {
        ints.push_back(std::shared_ptr<PotentialInt4C>(new PotentialInt4C(auxiliary_,zero,primary_,primary_,0,a_,b_,w_)));
    }

    // => Schwarz Shell Pairs <= //

    const std::vector<std::pair<int,int>> shell_pairs = sieve_->shell_pairs();

    // ==> Master Loop <== //
    
    for (size_t Atask = 0; Atask < Ashell_tasks.size() - 1; Atask++) {

        // => Shell Task Indexing <= //

        size_t Astart  = Ashell_tasks[Atask];
        size_t Astop   = Ashell_tasks[Atask+1];
        size_t Asize   = Astop - Astart;
        size_t astart  = auxiliary_->shell(Astart).function_index();
        size_t astop   = (Astop == auxiliary_->nshell() ? auxiliary_->nfunction() : auxiliary_->shell(Astop).function_index()); 
        size_t asize   = astop - astart;
        size_t PQsize  = shell_pairs.size();
        size_t APQsize = Asize * PQsize;

        // => Integrals <= //
        
        #pragma omp parallel for schedule(dynamic)
        for (size_t APQtask = 0; APQtask < APQsize; APQtask++) {
            size_t A  = APQtask / PQsize + Astart;
            size_t PQ = APQtask % PQsize;
            size_t P = shell_pairs[PQ].first; 
            size_t Q = shell_pairs[PQ].second; 
            int nA = auxiliary_->shell(A).nfunction();
            int nP = primary_->shell(P).nfunction();
            int nQ = primary_->shell(Q).nfunction();
            size_t oA = auxiliary_->shell(A).function_index();
            size_t oP = primary_->shell(P).function_index();
            size_t oQ = primary_->shell(Q).function_index();
            int t = omp_get_thread_num();
            ints[t]->compute_shell(A,0,P,Q);
            double* buffer = ints[t]->buffer();
            for (int a = 0; a < nA; a++) {
            for (int p = 0; p < nP; p++) {
            for (int q = 0; q < nQ; q++) {
                Ap[(a + oA - astart) * nbf * nbf + (p + oP) * nbf + (q + oQ)] =
                Ap[(a + oA - astart) * nbf * nbf + (q + oQ) * nbf + (p + oP)] =
                (*buffer++);
            }}}
        }

        // => Orbital Transform <= //
        
        for (const std::vector<std::string>& task : tasks) {
            
            // > First Half-Transform < //

            const Tensor& Cl = Cls_.at(task[0]);
            size_t n1 = Cl.dim(1); 
            Api.gemm(Apq,Cl,false,false,asize*nbf,n1,nbf,nbf,n1,n1,0,0,0,1.0,0.0);
            
            for (const std::string& key : task) {
                
                // > Second Half-Transform < //

                const Tensor& Cr = Crs_.at(key); 
                size_t n2 = Cr.dim(1); 
                const std::string& stripe = stripings_.at(key);
                Tensor& target = targets[key];
                if (stripe == "lrQ" || stripe == "Qlr") {
                    #pragma omp parallel for
                    for (size_t a = 0; a < asize; a++) {
                        Aia.gemm(Api,Cr,true,false,n1,n2,nbf,n1,n2,n2,a*nbf*n1,n2,a*n1*n2,1.0,0.0);
                    }
                    target({{astart,astop},{0,n1*n2}}) = Aia({{0,asize},{0,n1*n2}});
                } else {
                    #pragma omp parallel for
                    for (size_t a = 0; a < asize; a++) {
                        Aia.gemm(Cr,Api,true,false,n2,n1,nbf,n2,n1,n1,n2,a*nbf*n1,a*n2*n1,1.0,0.0);
                    }
                    target({{astart,astop},{0,n2*n1}}) = Aia({{0,asize},{0,n2*n1}});
                }
            }
        }
    }

    return targets;
}
std::map<std::string, tensor::Tensor> MODFERI::fit(const std::map<std::string, tensor::Tensor>& Aias) const
{
    // => Metric Power Merging <= //

    std::vector<std::vector<std::string>> tasks;
    for (size_t kind = 0; kind < keys_.size(); kind++) {
        std::string key = keys_[kind];
        bool found = true;
        for (size_t tind = 0; tind < tasks.size(); tind++) {
            if (powers_.at(key) == powers_.at(tasks[tind][0])) {
                tasks[tind].push_back(key);
                found = true;
                break;
            }    
        }
        if (!found) tasks.push_back({key});
    }

    // => Targets <= //

    std::map<std::string, Tensor> targets;
    
    // ==> Master Loop <== //

    for (const std::vector<std::string>& task : tasks) {

        // => Metric Power <= //
        
        Tensor J = metric_power_core(powers_.at(task[0]),metric_condition_);
        size_t naux = J.dim(0);

        // => Task Fitting <= //

        for (const std::string& key : task) {
            
            const Tensor& Aia = Aias.at(key);
            const Tensor& Cl = Cls_.at(key);
            const Tensor& Cr = Crs_.at(key);
            const std::string& stripe = stripings_.at(key);
            size_t n1;
            size_t n2;
            if (stripe == "lrQ" || stripe == "Qlr") {
                n1 = Cl.dim(1);
                n2 = Cr.dim(1);
            } else {
                n2 = Cl.dim(1);
                n1 = Cr.dim(1);
            }

            if (stripe == "lrQ" || stripe == "rlQ") {
                targets[key] = Tensor::build(kDisk, key, {naux,n1,n2});
                size_t rem = doubles() - naux * naux;
                size_t max1 = rem / (n2 * naux);
                max1 = std::min(max1,n1);
                if (max1 < 1) throw std::runtime_error("MODFERI: out of memory");

                Tensor L = Tensor::build(kCore,"L",{naux,max1*n2});
                Tensor R = Tensor::build(kCore,"R",{max1,n2,naux});

                for (size_t i1 = 0; i1 < n1; i1+=max1) {
                    size_t size1 = (i1 + max1 >= n1 ? n1 - i1 : max1);
                    L({{0,naux},{0,size1*n2}}) = Aia({{0,naux},{(i1)*n2,(i1 + size1)*n2}});
                    R.gemm(L,J,true,false,size1*n2,naux,naux,max1*n2,naux,naux,0,0,0,1.0,0.0);
                    targets[key]({{i1,i1+size1},{0,n2},{0,naux}}) = R({{0,size1},{0,n2},{0,naux}});
                }
            } else {
                targets[key] = Tensor::build(kDisk, key, {n1,n2,naux});
                size_t rem = doubles() - naux * naux;
                size_t max1 = rem / (n2 * naux);
                max1 = std::min(max1,n1);
                if (max1 < 1) throw std::runtime_error("MODFERI: out of memory");

                Tensor L = Tensor::build(kCore,"L",{naux,max1*n2});
                Tensor R = Tensor::build(kCore,"R",{naux,max1*n2});

                for (size_t i1 = 0; i1 < n1; i1+=max1) {
                    size_t size1 = (i1 + max1 >= n1 ? n1 - i1 : max1);
                    L({{0,naux},{0,size1*n2}}) = Aia({{0,naux},{(i1)*n2,(i1 + size1)*n2}});
                    R.gemm(L,J,true,false,size1*n2,naux,naux,max1*n2,naux,naux,0,0,0,1.0,0.0);
                    targets[key]({{0,naux},{(i1)*n2,(i1 + size1)*n2}}) = R({{0,naux},{0,size1*n2}});
                }
            }
        }
    }

    return targets;
}

} // namespace libgaussian
