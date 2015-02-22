#ifndef SCHWARZ_H
#define SCHWARZ_H

#include <cstddef>
#include <vector>
#include <memory>

namespace libgaussian {

class SBasisSet;

/**!
 * Class SchwarzSieve provides a truncation of the pair space based on the
 * exponential decay of CGTOs, as elucidated by the Cauchy-Schwarz Inequality
 *
 * => Math <=
 *
 *  Regarding ERI's (or any other SPD operator), The Cauchy-Schwarz Inequality
 *  states
 *
 *  |(pq|rs)| <= \sqrt{(pq|pq)(rs|rs)} <= \sqrt{(pq|pq)} \sqrt{(rs|rs)_max} <= T
 *
 **/
class SchwarzSieve {

public:
    SchwarzSieve(
        const std::shared_ptr<SBasisSet>& basis1,
        const std::shared_ptr<SBasisSet>& basis2,
        double cutoff,
        double a = 1.0,
        double b = 0.0,
        double w = 0.0);

    bool symmetric() const { return basis1_ == basis2_; }
    const std::shared_ptr<SBasisSet>& basis1() const { return basis1_; }
    const std::shared_ptr<SBasisSet>& basis2() const { return basis2_; }
    double cutoff() const { return cutoff_; }
    double a() const { return a_; }
    double b() const { return b_; }
    double w() const { return w_; }
    
    const std::vector<std::pair<int,int>>& shell_pairs() const { return shell_pairs_; }
    //const std::vector<std::vector<int>>& shell_to_shell() const { return shell_to_shell_; }

    //const std::vector<std::pair<int,int> >& function_pairs() const { return function_pairs_; }
    //const std::vector<std::vector<int> >& function_to_function() const { return function_to_function_; }

    void print(FILE* fh = stdout) const;
    
    /**!
     * Return the maximum possible value for a shell of type (PQ|PQ)
     *
     * NOTE: (PQ|RS) is not bounded by (PQ|PQ)
     *
     * @param P the shell index in basis1
     * @param Q the shell index in basis2
     * @return upper bound on magnitude of (PQ|PQ) integrals
     **/
    double shell_estimate_PQPQ(
        size_t P,
        size_t Q) const;

    /**!
     * Return the maximum possible value for a shell of type (PQ|RS), which is
     * given by the Cauchy-Schwarz bound \sqrt{(PQ|PQ)(RS|RS)}
     *
     * @param P the shell index in basis1 in coordinate 1
     * @param Q the shell index in basis2 in coordinate 1
     * @param R the shell index in basis1 in coordinate 2
     * @param S the shell index in basis2 in coordinate 2
     * @return upper bound on magnitude of (PQ|RS) integrals
     **/
    double shell_estimate_PQRS(
        size_t P,
        size_t Q,
        size_t R,
        size_t S) const;

private:
    std::shared_ptr<SBasisSet> basis1_;
    std::shared_ptr<SBasisSet> basis2_;
    double cutoff_;
    double a_;
    double b_;
    double w_;

    /// The maximum ERIs for shells stored in rectangular order
    double overall_max_;
    std::vector<double> shell_maxs_;
    std::vector<std::pair<int,int>> shell_pairs_;
    //std::vector<std::vector<int>> shell_to_shell_;
    
    void build_integrals();
    void build_sieve();
    
};    

} // namespace libgaussian

#endif
