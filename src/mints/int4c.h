#ifndef INT4C_H
#define INT4C_H

#include <cstddef>
#include <memory>
#include <core/basisset.h>

#include "fjt.h"

#include <libint/libint.h>

namespace libgaussian {

/**!
 * Class Int4C provides a common interface for the low-level computation of
 * four-center Gaussian integrals, including ERIs. Integral derivatives are
 * also covered under this scope.
 *
 * => Example Use <=
 *
 * Let us assume we want to compute the ERIs and their derivatives.
 *
 *  /// Construct a PotentialInt4C object for our basis sets and derivative level
 *  PotentialInt4C vints(basis1, basis2, basis3, basis4, 1);
 *  /// Get a pointer to the buffer where the integrals will be placed
 *  double* buffer = vints.buffer();
 *  /// Compute the ERI for the 3-th, 4-th, 5-th, and 6-th shells in each center
 *  vints.compute_shell0(3,4,5,6);
 *  /// Use the integrals layed out in buffer as described in the appropriate
 *  /// subclass below
 *  ...
 *  /// Compute the ERI derivatives for the same shells
 *  vints.compute_shell1(3,4,5,6);
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
class Int4C {

public:

    // => Constructors <= //

    /**!
     * Verbatim constructor, copies fields below
     **/
    Int4C(
        const std::shared_ptr<SBasisSet>& basis1,
        const std::shared_ptr<SBasisSet>& basis2,
        const std::shared_ptr<SBasisSet>& basis3,
        const std::shared_ptr<SBasisSet>& basis4,
        int deriv = 0);

    /// Default constructor, no initialization (except nullptr buffers)
    Int4C();

    /// Virtual destructor (deletes buffers if needed)
    virtual ~Int4C();

    // => Accessors <= //

    /// Basis set for center 1
    const std::shared_ptr<SBasisSet>& basis1() const { return basis1_; }
    /// Basis set for center 2
    const std::shared_ptr<SBasisSet>& basis2() const { return basis2_; }
    /// Basis set for center 3
    const std::shared_ptr<SBasisSet>& basis3() const { return basis3_; }
    /// Basis set for center 4
    const std::shared_ptr<SBasisSet>& basis4() const { return basis4_; }
    /// Maximum derivative level enabled
    int deriv() const { return deriv_; }

    /// Buffer of output integrals or integral derivatives (you do not own this)
    double* buffer() const { return buffer1_; }

    /// Single maximum angular momentum present across the basis sets
    int max_am() const;
    /// Total maximum angular momentum across the basis sets
    int total_am() const;
    /// Return the chunk size (max_ncart1 x max_ncart2 x max_ncart3 x max_ncart4)
    size_t chunk_size() const;

    // => Low-Level Computers <= //

    /// Compute the integrals (throws if not implemented)
    void compute_shell(
        size_t shell1,
        size_t shell2,
        size_t shell3,
        size_t shell4);
    /// Compute the integral derivatives (throws if not implemented)
    void compute_shell1(
        size_t shell1,
        size_t shell2,
        size_t shell3,
        size_t shell4);
    /// Compute the integral second derivatives (throws if not implemented)
    void compute_shell2(
        size_t shell1,
        size_t shell2,
        size_t shell3,
        size_t shell4);

    /// Compute the integrals (throws if not implemented)
    virtual void compute_shell(
        const SGaussianShell& sh1,
        const SGaussianShell& sh2,
        const SGaussianShell& sh3,
        const SGaussianShell& sh4);
    /// Compute the first derivatives (throws if not implemented)
    virtual void compute_shell1(
        const SGaussianShell& sh1,
        const SGaussianShell& sh2,
        const SGaussianShell& sh3,
        const SGaussianShell& sh4);
    /// Compute the  second derivatives (throws if not implemented)
    virtual void compute_shell2(
        const SGaussianShell& sh1,
        const SGaussianShell& sh2,
        const SGaussianShell& sh3,
        const SGaussianShell& sh4);

protected:

    std::shared_ptr<SBasisSet> basis1_;
    std::shared_ptr<SBasisSet> basis2_;
    std::shared_ptr<SBasisSet> basis3_;
    std::shared_ptr<SBasisSet> basis4_;
    int deriv_;
    /// Buffer for integrals, target (subclass allocates, super destroys)
    double* buffer1_;
    /// Buffer for CO->SO transformations (subclass allocates, super destroys)
    double* buffer2_;
    /// Internal CO->SO transformation information
    std::vector<SAngularMomentum> am_info_;

    Libint_t libint_;

};

/**!
 * Class PotentialInt4C computes two-electron potential integrals (ERIs) of the
 * form:
 *
 *  I_pqrs = \iint_{\mathbb{R}^6}
 *           \mathrm{d}^3 r_1
 *           \mathrm{d}^3 r_2
 *           \phi_p^1 \phi_q^1
 *           o(r_12)
 *           \phi_r^2 \phi_s^2
 *
 * This object supports the construction of integrals for generalized LRC
 * interaction kernels of the form:
 *
 *  o(r_12) = a / r_12 + b erf(w r_12) / r_12
 *
 * LRC-type objects are used in an identical manner as standard types.
 * The default behavior is the usual o(r_12) = 1 / r_12
 **/
class PotentialInt4C final : public Int4C {

public:
    PotentialInt4C(
        const std::shared_ptr<SBasisSet>& basis1,
        const std::shared_ptr<SBasisSet>& basis2,
        const std::shared_ptr<SBasisSet>& basis3,
        const std::shared_ptr<SBasisSet>& basis4,
        int deriv = 0,
        double a = 1.0,
        double b = 0.0,
        double w = 0.0);
    virtual ~PotentialInt4C();

    double a() const { return a_; }
    double b() const { return b_; }
    double w() const { return w_; }

    void compute_shell(
        const SGaussianShell& sh1,
        const SGaussianShell& sh2,
        const SGaussianShell& sh3,
        const SGaussianShell& sh4) override;
    void compute_shell1(
        const SGaussianShell& sh1,
        const SGaussianShell& sh2,
        const SGaussianShell& sh3,
        const SGaussianShell& sh4) override;
    void compute_shell2(
        const SGaussianShell& sh1,
        const SGaussianShell& sh2,
        const SGaussianShell& sh3,
        const SGaussianShell& sh4) override;

protected:

    double a_;
    double b_;
    double w_;

    Fjt *fjt_;

};

} // namespace libgaussian

#endif
