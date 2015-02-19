#ifndef DFERI_H
#define DFERI_H

#include <cstddef>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <tensor.h>

namespace libgaussian {

class SBasisSet;
class SchwarzSieve;

/**!
 * Class DFERI is a gateway to density fitted computations of various types
 *
 * Typically, one builds a specific DFERI subclass, changes some knobs,
 * possibly queues a set of tasks up, and then calls a compute method to do one
 * or more task at once.
 * 
 * These objects are all internally threaded, and should be called from a
 * master thread for optimal performance.
 *
 * These objects support the construction of DF integrals for generalized LRC
 * interaction kernels of the form:
 *
 *  o(r_12) = a / r_12 + b erf(w r_12) / r_12
 *
 * LRC-type DF objects are used in an identical manner as standard DF types.
 * The default behavior is the usual o(r_12) = 1 / r_12
 **/
class DFERI {

public:

    // => Constructors <= //

    /**!
     * Verbatim constructor, copies fields below
     **/
    DFERI(
        const std::shared_ptr<SBasisSet>& primary,
        const std::shared_ptr<SBasisSet>& auxiliary,
        const std::shared_ptr<SchwarzSieve>& sieve);

    /// Default constructor, no initialization
    DFERI() {}

    /// Virtual destructor
    virtual ~DFERI() {}

    // => Accessors <= //

    /// The primary basis set (size np)
    const std::shared_ptr<SBasisSet>& primary() const { return primary_; }
    /// The auxilairy basis set (size nQ)
    const std::shared_ptr<SBasisSet>& auxiliary() const { return auxiliary_; }
    /// The SchwarzSieve to guide significant shell pairs in sieved triangular indexing
    const std::shared_ptr<SBasisSet>& sieve() const { return sieve_; }
    /// The allowed memory usage in doubles
    size_t doubles() const { return doubles_; }
    /// The prefactor of 1/r_12 (default 1.0)
    double a() const { return a_; }
    /// The prefactor of in erf(w r_12) to apply (default 0.0)
    double b() const { return b_; }
    /// The Ewald parameter in erf(w r_12) / r_12 to apply (default 0.0)
    double w() const { return w_; }
    /// The maximum effective relative condition number to allow in the metric
    double metric_condition() const { return metric_condition_; }

    // => Setters <= //

    void set_doubles(size_t doubles) { doubles_ = doubles; }
    void set_a(double a) { a_ = a; }
    void set_b(double b) { b_ = b; }
    void set_w(double w) { w_ = w; }
    void set_metric_condition(double metric_condition) { metric_condition_ = metric_condition; }

    // => Methods <= //

    /**!
     * Compute the metric matrix (A|B)
     * @return the metric matrix as kCore
     **/
    Tensor metric() const;

    /**!
     * Compute the metric matrix (A|B)^power by eigendecomposition and
     * pseudoinversion
     * @param power the power to apply
     * @param condition the maximum relative condition number to keep in the
     *  metric
     * @return the metric matrix power as kCore
     **/
    Tensor metric_power(
        double power = -1.0/2.0, 
        double condition) const;

protected:

    std::shared_ptr<SBasisSet> primary_;
    std::shared_ptr<SBasisSet> auxiliary_;
    std::shared_ptr<SchwarzSieve> sieve_;

    size_t doubles_;
    double a_;
    double b_;
    double w_;
    double metric_condition_;
};

/**!
 * Class AODFERI produces fitted 3-index DF integrals in the AO basis, with
 * Schwarz sparsity and permutational symmetry built in by shell
 *
 * => Example Use Case <=
 *
 * Description:
 *  Get the AO-basis integrals on core for DF-SCF:
 * 
 * Definitions:
 *  3-index integral target - b_pq^Q is striped [Qpq] (nQ x npq, sieved
 *                            triangular indexing) 
 *
 *  b_pq^Q = (Q|A)^{-1/2} (A|pq)
 * 
 * Code:
 *  // Constructor
 *  AODFERI aodf(primary, auxiliary, sieve);
 *  ...
 *  // Knobs
 *  aodf.set_double(doubles);
 *  // Compute the 
 *  Tensor b = aodf.compute_ao_task_core(-1.0/2.0);
 *  // If needed, grab the SchwarzSieve for indexing
 *  std::shared_ptr<SchwarzSieve> sieve2 = aodf.sieve();
 *  // Do stuff with the tensor
 *  ...
 *
 * Notes:
 *  Whether to construct a kCore or kDisk Tensor is up to the user, if
 *  sufficient memory is available. The code will throw if a kCore tensor is
 *  requested without enough memory - there is no internal switch.
 **/
class AODFERI final : public DFERI {

public:

    // => Constructors <= //

    /**!
     * Verbatim constructor, copies fields below
     **/
    AODFERI(
        const std::shared_ptr<SBasisSet>& primary,
        const std::shared_ptr<SBasisSet>& auxiliary,
        const std::shared_ptr<SchwarzSieve>& sieve);

    /// Default constructor, no initialization
    AODFERI() {}

    // => Methods <= //

    /**!
     * Compute the AO-basis fitted DF integrals with the striping Q x pq, where
     * pq is sieved reduced triangular indexing (by shell pair) from the
     * SchwarzSieve object
     * @return the DF integrals as a CoreTensor
     **/
    Tensor compute_ao_task_core(double power = -1.0/2.0);
    /**!
     * Compute the AO-basis fitted DF integrals with the striping Q x pq, where
     * pq is sieved reduced triangular indexing (by shell pair) from the
     * SchwarzSieve object
     * @return the DF integrals as a DiskTensor
     **/
    Tensor compute_ao_task_disk(double power = -1.0/2.0);

protected:

};

/**!
 * Class MODFERI produces fitted 3-index DF integrals with orbital
 * transformations applied
 *
 * => Example Use Case <= 
 *
 * Description:
 *  Get the hole-particle (i->a) DF integrals for use in computing the DF-MP2
 *  OPDM vv block
 *
 * Definitions:
 *  3-index integral target - b_ia^Q is striped [iaQ] (ni x na x nQ) 
 *  orbitals for center 1   - C1_p^i is striped [pi] (np x ni)
 *  orbitals for center 2   - C2_q^a is striped [qa] (np x na)
 *  i <= a (for performance example)
 *
 *  b_ia^Q = (Q|A)^{-1/2} (A|pq) C1_p^i C2_q^a
 *
 * Code:
 *  // Constructor
 *  MODFERI modf(primary, auxiliary, sieve);
 *  ...
 *  // Knobs
 *  modf.set_double(doubles);
 *  ...
 *  // Queue task with key "biaQ"
 *  modf.add_mo_task("biaQ", C1, C2, -1.0/2.0, "lrQ");
 *  // Compute task(s) - Heavy operation
 *  std::map<std::string Tensor> results = modf.compute_mo_tasks();
 *  // Extract the result Tensor (kDisk) by key
 *  Tensor biaQ = results["biaQ"];
 *  // Do stuff with the tensor
 *  ...
 * 
 * => Example Use Case <= 
 *
 * Description:
 *  Get the hole-particle (i->a) *and* particle-hole (a->i) DF integrals for
 *  use in computing the DF-MP2 OPDM vv and oo blocks together
 *
 *  This example burns a few FLOPs (one could permute the former tensor to
 *  obtain the latter), but in practice you will probably not see the extra
 *  overhead for reasons discussed below
 *
 * Definitions:
 *  3-index integral target - b_ia^Q is striped [iaQ] (ni x na x nQ) 
 *  3-index integral target - b_ai^Q is striped [aiQ] (ni x na x nQ) 
 *  orbitals for center 1/2 - C1_p^i is striped [pi] (np x ni)
 *  orbitals for center 2/1 - C2_q^a is striped [qa] (np x na)
 *  i <= a (for performance example)
 *
 *  b_ia^Q = (Q|A)^{-1/2} (A|pq) C1_p^i C2_q^a
 *  b_ai^Q = (Q|A)^{-1/2} (A|pq) C2_p^a C1_q^i
 *
 * Code:
 *  // Constructor
 *  MODFERI modf(primary, auxiliary, sieve);
 *  ...
 *  // Knobs
 *  modf.set_double(doubles);
 *  ...
 *  // Queue task with key "biaQ"
 *  modf.add_mo_task("biaQ", C1, C2, -1.0/2.0, "lrQ");
 *  // Queue another taks with key "baiQ"
 *  modf.add_mo_task("baiQ", C1, C2, -1.0/2.0, "rlQ");
 *  // Compute task(s) - Heavy operation
 *  std::map<std::string Tensor> results = modf.compute_mo_tasks();
 *  // Extract the result Tensors (kDisk) by key
 *  Tensor biaQ = results["biaQ"];
 *  Tensor baiQ = results["baiQ"];
 *  // Do stuff with the tensors
 *  ...
 * 
 * Notes:
 *  In this example the three-index integrals (A|pq) and inverse metric
 *  (Q|A)^{-1/2} are computed only once across both tasks. Moreover, the code
 *  recognizes that the C1 transformation matrix is the same for both tasks (by
 *  checking the Tensor pointers), and only does one half-transformation to
 *  (A|iq). The second-half transform to (A|ia) and (A|ai) is where the paths
 *  diverge for the two tasks, as they have different permutations.* The metric
 *  application and disk permutation is also repeated twice. 
 *
 *  Note that the best performance is obtained when the smaller orbital space
 *  is used for C1. E.g., in the example above, the first-half transformation
 *  for both tasks scales as [O(ip^2Q)]. If we had instead specified the second
 *  task as,
 *
 *   modf.add_mo_task("baiQ", C2, C1, -1.0/2.0, "lrQ"); // Technically valid
 *
 *  We would get numerically the same result, but much worse performance, as
 *  this task's first-half transformation now scales as [O(ap^2Q)], a factor of
 *  a / i bigger than originally. Moreover, the first-half transformation must
 *  also be done separately for the two tasks with this specification.
 *
 *  *More technically, the code doesn't check for similarity in the second-half
 *  transform, as there is no performance gain to be obtained for unique tasks.
 **/
class MODFERI final : public DFERI {

public:
    
    // => Constructors <= //

    /**!
     * Verbatim constructor, copies fields below
     **/
    MODFERI(
        const std::shared_ptr<SBasisSet>& primary,
        const std::shared_ptr<SBasisSet>& auxiliary,
        const std::shared_ptr<SchwarzSieve>& sieve);

    /// Default constructor, no initialization
    MODFERI() {}

    // => Methods <= //

    /// Clear all tasks from this MODFERI (not needed on first use)
    void clear();

    /**!
     * Add a queued task to this MODFERI
     *
     * @param key - the key by which this task is known (also name of 3-index tensor)
     * @param Cl  - the first set of molecular orbitals, ordered np x nl
     * @param Cr  - the second set of molecule orbitals, ordered np x nr 
     * @param power - the power of metric to apply 
     * @param striping - the desired striping of the result tensor, a
     *  permutation string with three characters (l,r, and Q), one of:
     *  "lrQ", "rlQ", "Qlr", "Qrl". 
     *  "lQr" and "rQl" are NOT allowed
     *
     * NOTE: For performance reasons, it is highly advised to put the smaller
     * orbital space as Cl, and then use the striping parameter to obtain the
     * desired permutation.
     * 
     * NOTE: lrQ or rlQ stripings have the same cost, and are the fastest
     * stripings to produce. Qlr and Qrl stripings involve an additional
     * permutation of the finished disk tensors.
     *
     * NOTE: only references to Cl and Cr are kept - do not modify the data in
     * your copy of these tensors
     **/
    void add_mo_task(
        const std::string& key,
        const Tensor& Cl,
        const Tensor& Cr,
        double power = -1.0/2.0,
        const std::string& striping = "lrQ");
        
    /**!
     * Compute all queued tasks in this MODFERI
     *
     * @return map from key to Tensor (kDisk) with the resultant 3-index
     * tensors requested above
     **/
    std::map<std::string, Tensor> compute_mo_tasks();

protected: 

    std::vector<std::string> keys_;
    std::map<std::string, Tensor> Cls_;
    std::map<std::string, Tensor> Crs_;
    std::map<std::string, double> powers_;
    std::map<std::string, std::string> stripings_;

};

} // namespace libgaussian

#endif
