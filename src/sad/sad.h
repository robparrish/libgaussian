#ifndef SAD_H
#define SAD_H

#include <memory>
#include <ambit/tensor.h>

namespace lightspeed {

class SMolecule;
class SBasisSet;

class SAD {

public:

    SAD(
        const std::shared_ptr<SMolecule>& molecule,
        const std::shared_ptr<SBasisSet>& primary,
        const std::shared_ptr<SBasisSet>& minao);

    void print(FILE* fh = stdout) const;

    ambit::Tensor compute_C() const;
    ambit::Tensor compute_Ca() const;
    ambit::Tensor compute_Cb() const;

protected:

    std::shared_ptr<SMolecule> molecule_;
    std::shared_ptr<SBasisSet> primary_;
    std::shared_ptr<SBasisSet> minao_;

    /**!
     * Internal helper to compute C tensor for key in:
     *  C 
     *  CA
     *  CB
     **/
    ambit::Tensor compute_C_helper(const std::string& key) const;
    
    /**!
     * Compute the orbitals for the A-th atom assuming an alpha or beta charge
     * of Q, with the result projected to the primary basis set.
     *
     * This should be called twice for UHF (once for alpha, once for beta), or
     * once for RHF (with the average of alpha and beta occupations)
     *   
     * The algorithm attempts to fully occupy the core orbitals (the electron
     * configuration of the next-lowest noble gas atom), followed by fractional
     * occupation of the valence orbitals (the difference electron
     * configuration in going to the next highest noble gas atom). If there are
     * less orbitals than core functions, the core occupations are *uniformly*
     * scaled to match, and the valence orbitals are unoccupied.
     *
     * @param A the index of the A-th atom, for which to generate orbitals
     * @param Q the number of alpha or beta electrons to add to this atom
     *
     * @return kCore Tensor of size nbf(A) x nocc(A) [N/2] with fractional
     * orbitals for the atom, in the primary basis set.
     **/
    ambit::Tensor compute_atom(
        size_t A,
        double Q) const;

};


} // namespace lightspeed

#endif

