#ifndef SCHWARZ_H
#define SCHWARZ_H

#include <cstddef>
#include <memory>

namespace libgaussian {

class SBasisSet;

/**!
 **/
class SchwarzSieve {

public:
    SchwarzSieve(
        const std::shared_ptr<SBasisSet>& basis1,
        const std::shared_ptr<SBasisSet>& basis2,
        double cutoff);

    SchwarzSieve(
        const std::shared_ptr<SBasisSet>& basis,
        double cutoff);

    bool symmetric() const { return basis1 == basis2; }
    const std::shared_ptr<SBasisSet>& basis1() const { return basis1_; }
    const std::shared_ptr<SBasisSet>& basis2() const { return basis2_; }
    double cutoff() const { return cutoff_; }
    
    const std::vector<std::pair<size_t,size_t> >& shell_pairs() const { return shell_pairs_; }
    const std::vector<std:vector<size_t> >& shell_to_shell() const { return shell_to_shell_; }

    const std::vector<std::pair<size_t,size_t> >& function_pairs() const { return function_pairs_; }
    const std::vector<std:vector<size_t> >& function_to_function() const { return function_to_function_; }

    
    
private:
    std::shared_ptr<SBasisSet> basis1_;
    std::shared_ptr<SBasisSet> basis2_;
    double cutoff_;
    
};    

} // namespace libgaussian

#endif
