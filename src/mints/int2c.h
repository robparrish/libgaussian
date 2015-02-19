#ifndef INT2C_H
#define INT2C_H

#include <cstddef>
#include <memory>
#include <vector>
#include <core/basisset.h>
#include "osrecur.h"

namespace libgaussian {

class SMolecule;

/**!
 * Class Int2C provides a common interface for the low-level computation of
 * two-center Gaussian integrals, including overlap, dipole, quadrupole,
 * kinetic, and angular momentum integrals. Integral derivatives are also
 * covered under this scope.
 *
 * => Example Use <=
 *
 * Let us assume we want to compute dipole integrals and their derivatives.
 *
 *  /// Construct a DipoleInt2C object for our basis sets and derivative level
 *  DipoleInt2C dints(basis1, basis2, 1);
 *  /// Get a pointer to the buffer where the integrals will be placed
 *  double* buffer = dints.buffer();
 *  /// Set the dipole center to (1.0,2.0,3.0) au
 *  dints.set_x(1.0);
 *  dints.set_y(2.0);
 *  dints.set_z(3.0);
 *  /// Compute the dipole integrals for the 3-th and 4-th shells in each center
 *  dints.compute_shell(3,4);
 *  /// Use the integrals layed out in buffer as described in the appropriate
 *  /// subclass below
 *  ...
 *  /// Compute the dipole integral derivatives for the same shells
 *  dints.compute_shell1(3,4);
 *  /// Use the integrals layed out in buffer as described in the appropriate
 *  /// subclass below
 *  ...
 *
 * => Additional Notes <=
 *
 * These objects are not thread-safe due to internal scratch arrays! The best
 * policy is to make one object for each thread.
 *
 * - Rob Parrish, 17 February, 2015
 **/
class Int2C {

public:

    // => Constructors <= //

    /**!
     * Verbatim constructor, copies fields below
     **/
    Int2C(
        const std::shared_ptr<SBasisSet>& basis1,
        const std::shared_ptr<SBasisSet>& basis2,
        int deriv = 0);

    /// Default constructor, no initialization (except nullptr buffers)
    Int2C();

    /// Virtual destructor (deletes buffers if needed)
    virtual ~Int2C();

    // => Accessors <= //

    /// Basis set for center 1
    const std::shared_ptr<SBasisSet>& basis1() const { return basis1_; }
    /// Basis set for center 2
    const std::shared_ptr<SBasisSet>& basis2() const { return basis2_; }
    /// Maximum derivative level enabled
    int deriv() const { return deriv_; }
    /// Buffer of output integrals or integral derivatives (you do not own this)
    double* buffer() const { return buffer1_; }

    /// x center in au where properties are centered (defaults to 0.0)
    double x() const { return x_; }
    /// y center in au where properties are centered (defaults to 0.0)
    double y() const { return y_; }
    /// z center in au where properties are centered (defaults to 0.0)
    double z() const { return z_; }

    /// Single maximum angular momentum present across the basis sets
    int max_am() const;
    /// Total maximum angular momentum across the basis sets
    int total_am() const;
    /// Return the chunk size (max_ncart1 x max_ncart2)
    size_t chunk_size() const;

    // => Setters <= //

    void set_x(double x) { x_ = x; }
    void set_y(double y) { y_ = y; }
    void set_z(double z) { z_ = z; }

    // => Low-Level Computers <= //

    /// Compute the integrals (throws if not implemented)
    void compute_shell(
        size_t shell1,
        size_t shell2);
    /// Compute the integral derivatives (throws if not implemented)
    void compute_shell1(
        size_t shell1,
        size_t shell2);
    /// Compute the integral second derivatives (throws if not implemented)
    void compute_shell2(
        size_t shell1,
        size_t shell2);

    /// Compute the integrals (throws if not implemented)
    virtual void compute_shell(
        const SGaussianShell& sh1,
        const SGaussianShell& sh2);
    /// Compute the first derivatives (throws if not implemented)
    virtual void compute_shell1(
        const SGaussianShell& sh1,
        const SGaussianShell& sh2);
    /// Compute the  second derivatives (throws if not implemented)
    virtual void compute_shell2(
        const SGaussianShell& sh1,
        const SGaussianShell& sh2);

protected:

    std::shared_ptr<SBasisSet> basis1_;
    std::shared_ptr<SBasisSet> basis2_;
    int deriv_;
    /// Buffer for integrals, target (subclass allocates, super destroys)
    double* buffer1_;
    /// Buffer for CO->SO transformations (subclass allocates, super destroys)
    double* buffer2_;
    /// Internal CO->SO transformation information
    std::vector<SAngularMomentum> am_info_;

    double x_;
    double y_;
    double z_;

};

/**!
 * Class OverlapInt2C computes overlap integrals of the form:
 *
 *  S_pq = \int_{\mathbb{R}^3}
 *         \mathrm{d}^3 r_1
 *         \phi_p^1 \phi_q^1
 **/
class OverlapInt2C final : public Int2C {

public:
    OverlapInt2C(
        const std::shared_ptr<SBasisSet>& basis1,
        const std::shared_ptr<SBasisSet>& basis2,
        int deriv = 0);

    void compute_shell(
        const SGaussianShell& sh1,
        const SGaussianShell& sh2) override;
    void compute_shell1(
        const SGaussianShell& sh1,
        const SGaussianShell& sh2) override;
    void compute_shell2(
        const SGaussianShell& sh1,
        const SGaussianShell& sh2) override;


protected:

    ObaraSaikaTwoCenterRecursion recursion_;

};

/**!
 * Class DipoleInt2C computes overlap integrals of the form:
 *
 *  X_pq^x = \int_{\mathbb{R}^3}
 *           \mathrm{d}^3 r_1
 *           \phi_p^1 [x]_O \phi_q^1
 *
 * Here O denotes the property origin, which may be set in Int2C above. The
 * lexical ordering of dipole integrals is x,y,z
 *
 **/
class DipoleInt2C final : public Int2C {

public:
    DipoleInt2C(
        const std::shared_ptr<SBasisSet>& basis1,
        const std::shared_ptr<SBasisSet>& basis2,
        int deriv = 0);

    void compute_shell(
        const SGaussianShell& sh1,
        const SGaussianShell& sh2) override;
    void compute_shell1(
        const SGaussianShell& sh1,
        const SGaussianShell& sh2) override;
    void compute_shell2(
        const SGaussianShell& sh1,
        const SGaussianShell& sh2) override;


protected:

    ObaraSaikaTwoCenterRecursion recursion_;

};

/**!
 * Class QuadrupoleInt2C computes overlap integrals of the form
 *
 *  Q_pq^xy = \int_{\mathbb{R}^3}
 *            \mathrm{d}^3 r_1
 *            \phi_p^1 [xy]_O \phi_q^1
 *
 * Here O denotes the property origin, which may be set in Int2C above. The
 * lexical ordering of quadrupole integrals is xx,xy,xz,yy,yz,zz
 **/
class QuadrupoleInt2C final : public Int2C {

public:
    QuadrupoleInt2C(
        const std::shared_ptr<SBasisSet>& basis1,
        const std::shared_ptr<SBasisSet>& basis2,
        int deriv = 0);

    void compute_shell(
        const SGaussianShell& sh1,
        const SGaussianShell& sh2) override;
    void compute_shell1(
        const SGaussianShell& sh1,
        const SGaussianShell& sh2) override;
    void compute_shell2(
        const SGaussianShell& sh1,
        const SGaussianShell& sh2) override;


protected:

};


/**!
 * Class KineticInt2C computes kinetic energy integrals of the form
 *
 *  T_pq = \int_{\mathbb{R}^3}
 *         \mathrm{d}^3 r_1
 *         \phi_p^1 [-1/2 \nabla^2] \phi_q^1
 **/
class KineticInt2C final : public Int2C {

public:
    KineticInt2C(
        const std::shared_ptr<SBasisSet>& basis1,
        const std::shared_ptr<SBasisSet>& basis2,
        int deriv = 0);

    void compute_shell(
        const SGaussianShell& sh1,
        const SGaussianShell& sh2) override;
    void compute_shell1(
        const SGaussianShell& sh1,
        const SGaussianShell& sh2) override;
    void compute_shell2(
        const SGaussianShell& sh1,
        const SGaussianShell& sh2) override;

protected:

    ObaraSaikaTwoCenterRecursion recursion_;

};

/**!
 * Class PotentialInt2C computes nuclear potential energy integrals of the form
 *
 *  V_pq = \int_{\mathbb{R}^3}
 *         \mathrm{d}^3 r_1
 *         \phi_p^1 \sum_A [-Z_A / r_1A] \phi_q^1
 *
 * This object supports the construction of integrals for generalized LRC
 * interaction kernels of the form:
 *
 *  o(r_12) = a / r_12 + b erf(w r_12) / r_12
 *
 * LRC-type objects are used in an identical manner as standard types.
 * The default behavior is the usual o(r_12) = 1 / r_12
 **/
class PotentialInt2C final : public Int2C {

public:
    PotentialInt2C(
        const std::shared_ptr<SBasisSet>& basis1,
        const std::shared_ptr<SBasisSet>& basis2,
        int deriv = 0,
        double a = 1.0,
        double b = 0.0,
        double w = 0.0);

    double a() const { return a_; }
    double b() const { return b_; }
    double w() const { return w_; }

    /// x positions of the point charges in au
    std::vector<double>& xs() { return xs_; }
    /// y positions of the point charges in au
    std::vector<double>& ys() { return ys_; }
    /// z positions of the point charges in au
    std::vector<double>& zs() { return zs_; }
    /// Charges of the point charges in au
    std::vector<double>& Zs() { return Zs_; }

    /**!
     * Set a potential from a SMolecule object
     *
     * @param mol the molecule to set the charge field and indexing to
     * @param use_nuclear use the nuclear charges or total charges?
     **/
    void set_nuclear_potential(
        const std::shared_ptr<SMolecule>& mol,
        bool use_nuclear = true);

    void compute_shell(
        const SGaussianShell& sh1,
        const SGaussianShell& sh2) override;
    void compute_shell1(
        const SGaussianShell& sh1,
        const SGaussianShell& sh2) override;
    void compute_shell2(
        const SGaussianShell& sh1,
        const SGaussianShell& sh2) override;

protected:

    double a_;
    double b_;
    double w_;

    std::vector<double> xs_;
    std::vector<double> ys_;
    std::vector<double> zs_;
    std::vector<double> Zs_;

};


} // namespace libgaussian

#endif
