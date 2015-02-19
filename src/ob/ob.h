#ifndef ONEBODY_H
#define ONEBODY_H

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
 * Class OneBody is a gateway to the automatic, threaded, screened computation
 * of various one-electron potentials, including overlap, dipole, quadrupole,
 * kinetic, and nuclear potential integrals. Integral derivatives are also
 * covered under this scope.
 *
 * The class has the capacity to handle integrals for two different basis sets
 * (basis1 and basis2 below), but deriviates currently must come from the same
 * basis set (sorry!)
 *
 * Typically, one builds a specific OneBody subclass, changes some knobs, and
 * then generates one or more Tensors by calling compute routines
 * 
 * These objects are all internally threaded, and should be called from a
 * master thread for optimal performance.
 **/
class OneBody {

public:

    // => Constructors <= //

    /**!
     * Verbatim constructor, copies fields below
     **/
    OneBody(
        const std::shared_ptr<SBasisSet>& basis1,
        const std::shared_ptr<SBasisSet>& basis2,
        double integral_tolerance = 0.0);

    /// Default constructor, no initialization
    OneBody() {}

    /// Virtual destructor
    virtual ~OneBody();

    // => Accessors <= //

    /// Are the basis sets on center1 and center2 the same?
    bool is_symmetric() const { return basis1_ == basis2_; }
    /// Basis set for center 1
    const std::shared_ptr<SBasisSet>& basis1() const { return basis1_; }
    /// Basis set for center 2
    const std::shared_ptr<SBasisSet>& basis2() const { return basis2_; }
    /// Target cutoff below which to neglect integrals (implementation-specific)
    double integral_tolerance() const { return integral_tolerance_; }

    /// x center in au where properties are centered (defaults to 0.0)
    double x() const { return x_; }
    /// y center in au where properties are centered (defaults to 0.0)
    double y() const { return y_; }
    /// z center in au where properties are centered (defaults to 0.0)
    double z() const { return z_; }

    // => Setters <= //

    void set_x(double x) { x_ = x; }
    void set_y(double y) { y_ = y; }
    void set_z(double z) { z_ = z; }

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
        Tensor& S,
        double scale = 1.0);

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
        Tensor& T,
        double scale = 1.0);

     /**!
     * Compute the X (dipole) matrices:
     *
     *  X_pq^x += scale^x * 
     *            \int_{\mathbb{R}^3} 
     *            \mathrm{d}^3 r_1
     *            \phi_p^1 [x]_O \phi_q^1
     *
     * Use the set_x/y/z methods above to set the property origin
     * The ordering of dipoles is X, Y, Z
     * 
     * @param X a vector of Tensors of size np1 x np2 to add the results into
     * @param scale the scales of integrals to add into X
     **/
    virtual void compute_X(
        std::vector<Tensor>& Xs,
        const std::vector<double> scale = {1.0, 1.0, 1.0});

     /**!
     * Compute the Q (dipole) matrices:
     *
     *  Q_pq^xy += scale^x * 
     *             \int_{\mathbb{R}^3} 
     *             \mathrm{d}^3 r_1
     *             \phi_p^1 [xy]_O \phi_q^1
     *
     * Use the set_x/y/z methods above to set the property origin
     * The ordering of quadrupoles is XX, XY, XZ, YY, YZ, ZZ
     * The quadrupoles have trace - they are simple cartesian quadrupoles
     * 
     * @param Q a vector of Tensors of size np1 x np2 to add the results into
     * @param scale the scales of integrals to add into Q
     **/
    virtual void compute_Q(
        std::vector<Tensor>& Qs,
        const std::vector<double> scale = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0});

     /**!
     * Compute the V (nuclear potental energy) matrix:
     *
     *  T_pq += scale * 
     *          \int_{\mathbb{R}^3} 
     *          \mathrm{d}^3 r_1
     *          \phi_p^1 [-1/2 \nabla^2] \phi_q^1
     * 
     * @param T a Tensor of size np1 x np2 to add the results into
     * @param scale the scale of integrals to add into T
     **/
    virtual void compute_V(
        Tensor& V,
        const std::shared_ptr<Molecule> mol,
        double scale = 1.0,
        bool use_nuclear = true,
        double a = 1.0,
        double b = 0.0,
        double w = 0.0);

    virtual void compute_V(
        Tensor& V,
        const std::vector<double>& x,
        const std::vector<double>& y,
        const std::vector<double>& z,
        const std::vector<double>& Z,
        double a = 1.0,
        double b = 0.0,
        double w = 0.0,
        double scale = 1.0);

    virtual void compute_ESP(
        const Tensor& D,
        std::vector<double>& V,
        const std::vector<double>& x,
        const std::vector<double>& y,
        const std::vector<double>& z,
        double a = 1.0,
        double b = 0.0,
        double w = 0.0,
        double scale = 1.0);
         
    // > Gradients < //

    virtual Tensor computeS1(
        const Tensor& D,
        Tensor& Sgrad,
        double scale = 0);
    virtual void compute_T1(
        const Tensor& D,
        Tensor& T,
        double scale = 1.0);
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
    
protected:

    std::shared_ptr<SBasisSet> basis1_;
    std::shared_ptr<SBasisSet> basis2_;
    double integral_tolerance_;

};

/**!
 * Class DirectOneBody provides a OneBody implementation using the standard
 * integrals in libmints to fill CoreTensor objects
 **/
class DirectOneBody final : public OneBody {

public:

    // => Constructors <= //

    /**!
     * Verbatim constructor, copies fields below
     **/
    DirectOneBody(
        const std::shared_ptr<SBasisSet>& basis1,
        const std::shared_ptr<SBasisSet>& basis2,
        double integral_tolerance = 0.0);

    /// Default constructor, no initialization
    DirectOneBody() {}

    // => Methods <= //


protected:

};

} // namespace libgaussian

#endif
