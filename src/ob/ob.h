#ifndef ONEBODY_H
#define ONEBODY_H

#include <cstddef>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <ambit/tensor.h>

namespace lightspeed {

class SMolecule;
class SBasisSet;
class SchwarzSieve;

/**!
 * Class OneBody is a gateway to the automatic, threaded, screened computation
 * of various one-electron potentials, including overlap, dipole, quadrupole,
 * kinetic, and nuclear potential integrals. Integral derivatives are also
 * covered under this scope.
 *
 * The class has the capacity to handle integrals for two different basis sets
 * (basis1 and basis2 below), but derivatives currently must come from the same
 * basis set (sorry!)
 *
 * These objects are all internally threaded, and should be called from a
 * master thread for optimal performance.
 **/
class OneBody {

public:

    // => Constructors <= //

    /**!
     * The easist way to construct as OneBody object is 
     * to specify a SchwarzSieve object which contains basis1 and basis2, and
     * also describes the significant parts of the pair space
     **/
    OneBody(const std::shared_ptr<SchwarzSieve>& sieve);

    /// Default constructor, no initialization
    OneBody() {}

    /// Virtual destructor
    virtual ~OneBody() {}

    // => Accessors <= //

    /// Are the basis sets on center1 and center2 the same?
    bool is_symmetric() const { return basis1_ == basis2_; }
    /// Schwarz sieve object describing the significant pair space
    const std::shared_ptr<SchwarzSieve>& sieve() const { return sieve_; }
    /// Basis set for center 1
    const std::shared_ptr<SBasisSet>& basis1() const { return basis1_; }
    /// Basis set for center 2
    const std::shared_ptr<SBasisSet>& basis2() const { return basis2_; }

    // => Methods <= //

    // > Integrals < //

    /**
     * Compute the S (overlap) matrix:
     *
     *  S_pq += scale * 
     *          \int_{\mathbb{R}^3} 
     *          \mathrm{d}^3 r_1
     *          \phi_p^1 \phi_q^1
     * 
     * @param S a Tensor of size np1 x np2 to add the results into
     * @param scale the scale of integrals to add into S
     **/
    virtual void compute_S(
        ambit::Tensor& S,
        double scale = 1.0) const;

     /**!
     * Compute the T (kinetic energy) matrix:
     *
     *  T_pq += scale * 
     *          \int_{\mathbb{R}^3} 
     *          \mathrm{d}^3 r_1
     *          \phi_p^1 [-1/2 \nabla^2] \phi_q^1
     * 
     * @param T a Tensor of size np1 x np2 to add the results into
     * @param scale the scale of integrals to add into T
     **/
    virtual void compute_T(
        ambit::Tensor& T,
        double scale = 1.0) const;

     /**!
     * Compute the X (dipole) matrices:
     *
     *  X_pq^x += scale^x * 
     *            \int_{\mathbb{R}^3} 
     *            \mathrm{d}^3 r_1
     *            \phi_p^1 [x]_O \phi_q^1
     *
     * The ordering of dipoles is X, Y, Z
     * 
     * @param X a vector of Tensors of size np1 x np2 to add the results into
     * @param scale the scales of integrals to add into X
     **/
    virtual void compute_X(
        std::vector<ambit::Tensor>& Xs,
        const std::vector<double>& origin = {0.0, 0.0, 0.0},
        const std::vector<double>& scale = {1.0, 1.0, 1.0}) const;

     /**!
     * Compute the Q (dipole) matrices:
     *
     *  Q_pq^xy += scale^x * 
     *             \int_{\mathbb{R}^3} 
     *             \mathrm{d}^3 r_1
     *             \phi_p^1 [xy]_O \phi_q^1
     *
     * The ordering of quadrupoles is XX, XY, XZ, YY, YZ, ZZ
     * The quadrupoles have trace - they are simple cartesian quadrupoles
     * 
     * @param Q a vector of Tensors of size np1 x np2 to add the results into
     * @param scale the scales of integrals to add into Q
     **/
    virtual void compute_Q(
        std::vector<ambit::Tensor>& Xs,
        const std::vector<double>& origin = {0.0, 0.0, 0.0},
        const std::vector<double>& scale = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0}) const;

     /**!
     * Compute the V (nuclear potental energy) matrix:
     *
     *  V_pq += scale * 
     *          \int_{\mathbb{R}^3} 
     *          \mathrm{d}^3 r_1
     *          \phi_p^1 [-Z_A / r_1A] \phi_q^1
     *
     * This routine can compute the potental matrix with an Ewald-type
     * interaction operator of the form:
     *
     *  a / r_12 + b erf(w r_12) / r_12
     *
     * Note: this object uses the conventional definition of these integrals,
     * in which positive Zs correspond to positive charges, which attract to
     * electrons, resulting in negative matrix elements for positive  Zs.
     * 
     * @param V a Tensor of size np1 x np2 to add the results into
     * @param xs the x positions of the point charges in au
     * @param ys the y positions of the point charges in au
     * @param zs the z positions of the point charges in au
     * @param Zs the magnitudes of the point charges in au
     * @param a the scale of the usual 1/r_12 interaction operator
     * @param b the scale of the Ewald erf(w r_12)/r_12 interaction operator
     * @param w the Ewald range-separation parameter
     * @param scale the scale of integrals to add into V
     **/
    virtual void compute_V(
        ambit::Tensor& V,
        const std::vector<double>& xs,
        const std::vector<double>& ys,
        const std::vector<double>& zs,
        const std::vector<double>& Zs,
        double a = 1.0,
        double b = 0.0,
        double w = 0.0,
        double scale = 1.0) const;

     /**!
     * Compute the V (nuclear potental energy) matrix:
     *
     *  V_pq += scale * 
     *          \int_{\mathbb{R}^3} 
     *          \mathrm{d}^3 r_1
     *          \phi_p^1 [-Z_A / r_1A] \phi_q^1
     *
     * This routine provides a convenient way to get the potential matrix for a
     * molecular source. In reality, this calls the more general compute_V
     * above
     *
     * This routine can compute the potental matrix with an Ewald-type
     * interaction operator of the form:
     *
     *  a / r_12 + b erf(w r_12) / r_12
     * 
     * @param V a Tensor of size np1 x np2 to add the results into
     * @param mol the molecule providing point charges {<Z_A,r_1A>}
     * @param use_nuclear use nuclear (Z) or total (Q) charges?
     * @param a the scale of the usual 1/r_12 interaction operator
     * @param b the scale of the Ewald erf(w r_12)/r_12 interaction operator
     * @param w the Ewald range-separation parameter
     * @param scale the scale of integrals to add into V
     **/
    virtual void compute_V_nuclear(
        ambit::Tensor& V,
        const std::shared_ptr<SMolecule> mol,
        bool use_nuclear = true,
        double a = 1.0,
        double b = 0.0,
        double w = 0.0, 
        double scale = 1.0) const;

    // > Gradients < //

    /**!
     * Compute the contraction of the overlap integral derivatives with the
     * (energy-weighted) density matrix:
     *
     *  Sgrad_Ai += scale *
     *              0.5 * (D_pq + D_qp) * 
     *              \partial_Ai
     *              \int_{\mathbb{R}^3} 
     *              \mathrm{d}^3 r_1
     *              \phi_p^1 \phi_q^1
     * 
     * Note that this method only works for symmetric OneBody objects, and that
     * the number of atoms in the gradient register must match that in the
     * basis set. 
     *
     * Note that this method performs the contraction with an effectively
     * symmetrized D matrix - you do not need to explicitly symmetrize D in
     * non-Hermetian and other weird methods
     *
     * @param D a Tensor of size np x np to contract against (energy-weighted
     * density matrix)
     * @param Sgrad a Tensor of size natom x 3 [0x,0y,0z; 1x,1y,1z, ...] to add
     * the resultant gradient contribution into
     * @param scale the scale of contraction to add into the result
     **/
    virtual void compute_S1(
        const ambit::Tensor& D,
        ambit::Tensor& Sgrad,
        double scale = 1.0) const;

    /**!
     * Compute the contraction of the kinetic integral derivatives with the
     * density matrix:
     *
     *  Sgrad_Ai += scale *
     *              0.5 * (D_pq + D_qp) * 
     *              \partial_Ai
     *              \int_{\mathbb{R}^3} 
     *              \mathrm{d}^3 r_1
     *              \phi_p^1 [-1/2 \nabla^2] \phi_q^1
     * 
     * Note that this method only works for symmetric OneBody objects, and that
     * the number of atoms in the gradient register must match that in the
     * basis set. 
     *
     * Note that this method performs the contraction with an effectively
     * symmetrized D matrix - you do not need to explicitly symmetrize D in
     * non-Hermetian and other weird methods
     *
     * @param D a Tensor of size np x np to contract against (one-particle
     * density matrix)
     * @param Tgrad a Tensor of size natom x 3 [0x,0y,0z; 1x,1y,1z, ...] to add
     * the resultant gradient contribution into
     * @param scale the scale of contraction to add into the result
     **/
    virtual void compute_T1(
        const ambit::Tensor& D,
        ambit::Tensor& Tgrad,
        double scale = 1.0) const;

#if 0 
    virtual void compute_V1(
        const Tensor& D,
        Tensor& V,
        const std::shared_ptr<Molecule> mol,
        bool use_nuclear = true,
        double a = 1.0,
        double b = 0.0,
        double w = 0.0,
        double scale = 1.0);
    
    // > Hessians < //
    
    virtual Tensor computeS2(
        const Tensor& D,
        Tensor& Sgrad,
        double scale = 0);
    virtual void compute_T2(
        const Tensor& D,
        Tensor& T,
        double scale = 1.0);
    virtual void compute_V2(
        const Tensor& D,
        Tensor& V,
        const std::shared_ptr<Molecule> mol,
        bool use_nuclear = true,
        double a = 1.0,
        double b = 0.0,
        double w = 0.0,
        double scale = 1.0);

#endif
    
protected:

    std::shared_ptr<SchwarzSieve> sieve_;
    std::shared_ptr<SBasisSet> basis1_;
    std::shared_ptr<SBasisSet> basis2_;

};

} // namespace lightspeed

#endif
