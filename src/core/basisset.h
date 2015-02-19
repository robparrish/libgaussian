#ifndef BASISSET_H
#define BASISSET_H

#include <cstddef>
#include <memory>
#include <vector>
#include <string>
#include "am.h"

namespace libgaussian {

/**!
 * Class SGaussianShell is a data container for a simple Gaussian basis shell
 *
 * Within a shell, the v-th function is defined as,
 *
 * \phi_{v} (\vec r_1) = AM_v (\vec r_1A) [\sum_{K}^{\nprim} c_K \exp(-e_K * r_1A^2)]
 *
 * Here, the shell is centered (usually on an atom) at the position \vec r_A =
 * <x_A,y_A,z_A>. The c_K and e_K are the primitive contraction coefficients
 * and primitive Gaussian exponents, respectively. AM_v (\vec r_1A) provides
 * the application of angular momentum to produce the v-th function with in the
 * shell, and depends on whether the shell is spherical or cartesian in nature.
 * See am.h for details on these two schemes. 
 *
 * => Notes on Contraction Coefficients <= 
 *
 * Note that the spherical and radial parts are direct-product separable. In
 * particular, there is no v-dependent normalization coefficient applied to the
 * radial part (thus the radial part is the same for all basis functions within
 * the shell). Consider a standard Gaussian basis set, e.g., cc-pVDZ for H
 * (with optimized contractions), specified by a G94 basis file:
 *
 * H     0
 * S   3   1.00
 *      13.0100000              0.0196850
 *       1.9620000              0.1379770
 *       0.4446000              0.4781480
 * S   1   1.00
 *       0.1220000              1.0000000
 * P   1   1.00
 *       0.7270000              1.0000000
 *           (e_K)                  (w_K)
 *  
 * Looking at the first shell (1s), we see that the shell is S (L = 0), has 3
 * primitive contractions (nprimitive = 3), and is to be normalized to 1.0 (the
 * 1.00 in the line "S 3 1.00"). The user has also provided the primitive
 * Gaussian exponents (first column, e_K) and primitive contraction *weights*
 * (second column, w_K). The e_K go directly into the basis shell object. The
 * w_K must first be normalized to produce the c_K contraction coefficients.
 * This is a two stage process. In the first stage, each primitive weight is
 * multiplied by the normalization of the corresponding normalized primitive
 * Gaussian:
 *
 * a_K = [\frac{\pi^{3/2} (2L - 1)!!}{2^L} \frac{1}{(e_K + e_K)^{L+3/2}} ]^{-1/2}
 *
 * Now the total contracted gaussian shell is normalized:
 *
 * N =  [\frac{\pi^{3/2} (2L - 1)!!}{2^L} \sum_{K,K'}^{\nprim} \frac{a_K a_K'}{(e_K + e_K')^{L+3/2}}]^{-1/2}
 *
 * And the normalization is applied to form each contraction coefficient:
 *
 * c_K = N a_K
 *
 * Note that if a shell is to be normalized to V instead of 1.0, the
 * normalization coefficient may be adjusted as,
 *
 * N <- \sqrt{V} N
 *
 * This code works only in terms of c_K - it is up to the user to specify c_K
 * as desired to express the underlying G94 file (or other source). 
 *
 * => Other Notes <= 
 *
 * In this library's convention a basis set may have mixed spherical and
 * cartesian angular momentum - this decision is made on a shell-by-shell
 * basis, and may be determined by using the is_spherical method below.
 *
 * A number of handy back-references to the atom index, shell index, function
 * index, and cartesian index within the containing basis set are also kept
 * with the Gaussian shell, and are used in placing data pertaining to the
 * shell into various arrays.
 *
 * - Rob Parrish, 17 February, 2015
 **/
class SGaussianShell {

public:

    // => Constructors <= //

    /// Verbatim constructor, fills fields below
    SGaussianShell(
        double x,
        double y,
        double z,
        bool is_spherical,
        int am,
        const std::vector<double>& cs,
        const std::vector<double>& es,
        size_t atom_index,
        size_t shell_index,
        size_t function_index,
        size_t cartesian_index) : 
        x_(x),
        y_(y),
        z_(z),
        is_spherical_(is_spherical),
        am_(am),
        cs_(cs),
        es_(es),
        atom_index_(atom_index),
        shell_index_(shell_index),
        function_index_(function_index),
        cartesian_index_(cartesian_index)
        {}

    /// Default constructor, no initialization
    SGaussianShell() {}

    // => Accessors <= //
    
    /// X position of shell (atomic center)
    double x() const { return x_; }
    /// Y position of shell (atomic center)
    double y() const { return y_; }
    /// Z position of shell (atomic center)
    double z() const { return z_; }
    
    /// Is this shell spherical or cartesian?
    bool is_spherical() const { return is_spherical_; } 

    /// Angular momentum of this shell
    int am() const { return am_; }   
    /// Number of functions in this shell (depends on is_spherical)
    size_t nfunction() const { return (is_spherical_ ? 2L*am_ + 1L : (am_ + 1L) * (am_ + 2L) / 2L); }
    /// Number of cartesian functions in this shell (before any spherical transformations)
    size_t ncartesian() const { return am_ * (am_ + 1L) / 2L; }
    
    /// Number of primitive Gaussians in this shell
    size_t nprimitive() const { return es_.size(); }
    /// The ind-th primitive contraction coefficient
    double c(size_t ind) const { return cs_[ind]; }
    /// The ind-th primitive Gaussian exponent
    double e(size_t ind) const { return es_[ind]; }
    /// The list of primitive contraction coefficients
    const std::vector<double>& cs() const { return cs_; }
    /// The list of primitive Gaussian exponents
    const std::vector<double>& es() const { return es_; }
 
    /// Index of the atom this shell is centered on within all atoms in this basis set
    size_t atom_index() const { return atom_index_; }
    /// Index of this shell within its containing basis set
    size_t shell_index() const { return shell_index_; }
    /// Starting function index of this shell within its containing basis set
    size_t function_index() const { return function_index_; }
    /// Starting cartesian index of this shell within its containing basis set
    size_t cartesian_index() const { return cartesian_index_; }

    // => Setters <= //

    void set_x(double x) { x_ = x; }
    void set_y(double y) { y_ = y; }
    void set_z(double z) { z_ = z; }
    void set_is_spherical(bool is_spherical) { is_spherical_ = is_spherical; }
    void set_am(int am) { am_ = am; }
    void set_cs(const std::vector<double>& cs) { cs_ = cs; }
    void set_es(const std::vector<double>& es) { es_ = es; }
    void set_atom_index(size_t atom_index) { atom_index_ = atom_index; }
    void set_shell_index(size_t shell_index) { shell_index_ = shell_index; }
    void set_function_index(size_t function_index) { function_index_ = function_index; }
    void set_cartesian_index(size_t cartesian_index) { cartesian_index_ = cartesian_index; }

private:
    double x_;
    double y_;
    double z_;
    bool is_spherical_;
    int am_;
    std::vector<double> cs_;
    std::vector<double> es_; 

    size_t atom_index_;
    size_t shell_index_;
    size_t function_index_;
    size_t cartesian_index_;
};

/**!
 * Class SBasisSet is a simple wrapper around a list of atom-affiliated
 * SGaussianShells, plus some utility functions.
 *
 * - Rob Parrish, 16 February, 2015
 **/ 
class SBasisSet {

public:

    // => Constructors <= //

    /**!
     * Verbatim constructor, copies fields below, unrolls the basis shell list,
     * performs indexing, and resets atom/shell/function/cartesian indices to
     * the order of this basis set.
     *
     * @param name the name of the SBasisSet
     * @param shells the SGaussianShells of this basis set, a list of
     * SGaussianShells for each atom
     **/
    SBasisSet(
        const std::string& name,
        const std::vector<std::vector<SGaussianShell>>& shells);

    /// Default constructor, no initialization
    SBasisSet() {}

    /// Build a null SBasisSet (for use in DF)
    std::shared_ptr<SBasisSet> zero_basis();

    // => Accessors <= //

    /// The basis set's name
    const std::string& name() const { return name_; }
    /// The ind-th shell
    const SGaussianShell& shell(size_t ind) const { return shells_[ind]; }
    /// The complete list of shells
    const std::vector<SGaussianShell>& shells() const { return shells_; } 
    /// The mapping from atom index and shell index within that atom to absolute shell index
    const std::vector<std::vector<size_t>>& atoms_to_shell_inds() const { return atoms_to_shell_inds_; }
    /// The SAngularMomentum objects op to max_am() for this basis set
    const std::vector<SAngularMomentum>& am_info() const { return am_info_; }

    /// Total number of atoms in this basis set
    size_t natom() const { return atoms_to_shell_inds_.size(); }
    /// Total number of shells in this basis set
    size_t nshell() const { return shells_.size(); }
    /// Total number of basis functions in this basis set
    size_t nfunction() const { return nfunction_; }
    /// Total number of cartesian functions in this basis set
    size_t ncartesian() const { return ncartesian_; }
    /// Total number of primitives in this basis set
    size_t nprimitive() const { return nprimitive_; }

    /// Does this molecule have any shells with spherical harmonics? 
    bool has_spherical() const;
    
    /// Maximum angular momentum across all shells
    int max_am() const;
    /// Maximum basis functions per shell across all shells
    size_t max_nfunction() const;
    /// Maximum cartesian functions per shell across all shells
    size_t max_ncartesian() const; 
    /// Maximum number of primitives across all shells
    size_t max_nprimitive() const;

private:

    std::string name_;
    std::vector<SGaussianShell> shells_; 
    std::vector<std::vector<size_t>> atoms_to_shell_inds_;
    std::vector<SAngularMomentum> am_info_;

    size_t nfunction_;
    size_t ncartesian_;
    size_t nprimitive_;
};

} // namespace libgaussian

#endif
