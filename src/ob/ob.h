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
     * @param V a Tensor of size np1 x np2 to add the results into
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

    virtual void compute_S1(
        const ambit::Tensor& D,
        ambit::Tensor& Sgrad,
        double scale = 1.0) const;

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
