#ifndef INT4C2_H
#define INT4C2_H

#include <cstddef>
#include <memory>
#include <vector>
#include <string>
#include <core/basisset.h>

namespace lightspeed {

class SBasisSet;
class SGaussianShell;
class SAngularMomentum;
class Interaction;

/**!
 * Class Int4C2 provides a common interface for the low-level computation of
 * four-center Gaussian integrals, including ERIs. Integral derivatives are
 * also covered under this scope.
 *
 * => Example Use <=
 *
 * Let us assume we want to compute the ERIs and their derivatives.
 *
 *  /// Construct a PotentialInt4C2 object for our basis sets and derivative level
 *  PotentialInt4C2 vints(basis1, basis2, basis3, basis4, 1);
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
class Int4C2
{

public:
    // => Constructors <= //

    virtual ~Int4C2() = default;

    // => Factory Constructors <= //

    static std::shared_ptr<Int4C2> coulomb(
        const std::shared_ptr<SBasisSet> &basis1,
        const std::shared_ptr<SBasisSet> &basis2,
        const std::shared_ptr<SBasisSet> &basis3,
        const std::shared_ptr<SBasisSet> &basis4,
        int deriv,
        const std::string& type = "LIBINT2");

    static std::shared_ptr<Int4C2> ewald(
        const std::shared_ptr<SBasisSet> &basis1,
        const std::shared_ptr<SBasisSet> &basis2,
        const std::shared_ptr<SBasisSet> &basis3,
        const std::shared_ptr<SBasisSet> &basis4,
        int deriv,
        double a,
        double b,
        double w,
        const std::string& type = "LIBINT2");

    static std::shared_ptr<Int4C2> f12(
            const std::shared_ptr<SBasisSet>& basis1,
            const std::shared_ptr<SBasisSet>& basis2,
            const std::shared_ptr<SBasisSet>& basis3,
            const std::shared_ptr<SBasisSet>& basis4,
            int deriv,
            double slater_exponent,
            const std::string& type = "LIBINT2");

    // => Accessors <= //

    /// Basis set for center 1
    const std::shared_ptr<SBasisSet> &basis1() const
    {
        return basis1_;
    }
    /// Basis set for center 2
    const std::shared_ptr<SBasisSet> &basis2() const
    {
        return basis2_;
    }
    /// Basis set for center 3
    const std::shared_ptr<SBasisSet> &basis3() const
    {
        return basis3_;
    }
    /// Basis set for center 4
    const std::shared_ptr<SBasisSet> &basis4() const
    {
        return basis4_;
    }
    /// The interaction operator for this Int4C2
    const std::shared_ptr<Interaction>& interaction() const
    {
        return interaction_;
    }
    /// Maximum derivative level enabled
    int deriv() const
    {
        return deriv_;
    }

    /// Buffer of output integrals or integral derivatives
    const std::vector<double> &data() const
    {
        return data1_;
    }
    /// Buffer of output integrals or integral derivatives (you do not own this)
    double *buffer() const
    {
        return buffer1_;
    }

    /// Should this object apply spherical transformations if present in the basis sets (defaults to true)
    bool is_spherical() const
    {
        return is_spherical_;
    }


    /// Single maximum angular momentum present across the basis sets
    int max_am() const;

    /// Total maximum angular momentum across the basis sets
    int total_am() const;

    /// Return the chunk size (max_ncart1 x max_ncart2 x max_ncart3 x max_ncart4)
    size_t chunk_size() const;

    // => Setters <= //

    void set_is_spherical(bool is_spherical)
    {
        is_spherical_ = is_spherical;
    }

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
    virtual void compute_quartet(
            const SGaussianShell &sh1,
            const SGaussianShell &sh2,
            const SGaussianShell &sh3,
            const SGaussianShell &sh4);

    /// Compute the first derivatives (throws if not implemented)
    virtual void compute_quartet1(
            const SGaussianShell &sh1,
            const SGaussianShell &sh2,
            const SGaussianShell &sh3,
            const SGaussianShell &sh4);

    /// Compute the  second derivatives (throws if not implemented)
    virtual void compute_quartet2(
            const SGaussianShell &sh1,
            const SGaussianShell &sh2,
            const SGaussianShell &sh3,
            const SGaussianShell &sh4);

protected:

    std::shared_ptr<SBasisSet> basis1_;
    std::shared_ptr<SBasisSet> basis2_;
    std::shared_ptr<SBasisSet> basis3_;
    std::shared_ptr<SBasisSet> basis4_;
    std::shared_ptr<Interaction> interaction_;
    int deriv_;

    std::vector<double> data1_;
    std::vector<double> data2_;
    double *buffer1_;
    double *buffer2_;

    bool is_spherical_;
    /// Internal CO->SO transformation information
    std::vector<SAngularMomentum> am_info_;

    /**!
     * Helper to apply spherical transformations to the cartesian integrals
     *
     * This should be called separately for each chunk
     *
     * Does not check is_spherical, the calling code is responsible for this
     *
     * Uses buffer2 for scratch space
     *
     * @param L1 angular momentum of shell1
     * @param L2 angular momentum of shell2
     * @param L3 angular momentum of shell3
     * @param L4 angular momentum of shell4
     * @param S1 perform transform for shell1
     * @param S2 perform transform for shell2
     * @param S3 perform transform for shell3
     * @param S4 perform transform for shell4
     * @param target pointer in buffer1 to start of chunk
     * @param scratch pointer (at least chunk size)
     * @result target is updated with the transformed integrals
     **/
    void apply_spherical(
            int L1,
            int L2,
            int L3,
            int L4,
            bool S1,
            bool S2,
            bool S3,
            bool S4,
            double *target,
            double *scratch);

    /**!
     * Verbatim constructor, copies fields
     **/
    Int4C2(
        const std::shared_ptr<SBasisSet> &basis1,
        const std::shared_ptr<SBasisSet> &basis2,
        const std::shared_ptr<SBasisSet> &basis3,
        const std::shared_ptr<SBasisSet> &basis4,
        const std::shared_ptr<Interaction> &interaction,
        int deriv);
};

} // namespace lightspeed

#endif
