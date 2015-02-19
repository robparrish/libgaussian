#ifndef AM_H
#define AM_H

#include <cstddef>
#include <vector>

namespace libgaussian {

/**!
 * Class SAngularMomentum describes the ordering and composition of cartesian
 * and spherical angular momentum applied to SGaussianShells in LibGaussian.
 * 
 * Typically, a user would generate a vector of SAngularMomentum, with one
 * entry each for [0, Lmax]. E.g., to treat up to G functions (L=4), the user
 * would say:
 *
 *  std::vector<SAngularMomentum> am_info(4+1);
 *  for (int L = 0; L <= 4; L++) {
 *     am_info[L] = SAngularMomentum(L);
 *  }
 *
 * Now the user may ask the SAngularMomentum for am = L for information on how
 * to build the cartesian or spherical angular momentum for the given shell.
 * Note: these objects are cheap to build, use wherever needed. 
 *
 * => Cartesian Functions <= 
 *
 * For cartesian functions, we adopt the following convention:
 *
 *  \phi_lmn (\vec r_1) = x_1A^l y_1A^m z_1A^n [\sum_{K}^{nprim} c_K \exp(-e_K r_1A^2)]
 * 
 * That is, angular momentum is applied independently in x,y,and z, and no
 * joint l,m,n-dependent normalization coefficient is applied. Note that this
 * implies that a cartesian d-shell which is normalized in xx, yy, and zz will
 * NOT be normalized in xy, xy, and yz. More generally, a diagonal function
 * (xx, yyy, zzzz, etc) will be normalized (if the underlying shell is
 * normalized), but non-diagonal functions (xy, xxy, xyzz, etc) will not be.  
 *
 * Now, all the remains is to specify a rule for the lexical ordering of l,m,n
 * tuples. These are given for convenience below in the l/m/n methods (and
 * plural variants thereof). Alternatively, users may find it more convenient
 * to use the standard algorithm for generating these:
 *
 *  for (int i = 0, index = 0; i <= L; i++) {
 *      int l = L - i;
 *      for (int j = 0; j <= i; j++, index++) {
 *          int m = i - j;
 *          int n = j;
 *      }
 *  }
 *
 * In English, this first adds angular momentum to l, then to m, then to n, in
 * a lexical manner.
 *
 * The ordering of cartesian prefactors for basis shells of up to G-type are
 * shown below:
 *
 * l/i  0    1    2    3    4    5    6    7    8    9   10   11   12   13   14
 * 0    1
 * 1    x    y    z
 * 2   xx   xy   xz   yy   yz   zz
 * 3  xxx  xxy  xxz  xyy  xyz  xzz  yyy  yyz  yzz  zzz
 * 4 xxxx xxxy xxxz xxyy xxyz xxzz xyyy xyyz xyzz xzzz yyyy yzzz yyzz yzzz zzzz
 * l/i  0    1    2    3    4    5    6    7    8    9   10   11   12   13   14
 *
 * => Spherical Functions <= 
 * 
 * To produce spherical functions, we invoke the standard real solid harmonics:
 *
 *  \phi_lm = S_lm (\vec r_1A) [\sum_{K}^{nprim} c_K \exp(- e_K r_1A^2)]
 *  = C_lm^lmn \phi_lmn (\vec r_1A)
 * 
 * The last expression demonstrates how the spherical basis functions are
 * produced as linear combinations of cartesian basis functions. The sparse
 * matrix C_lm^lmn provides the coefficients of the cartesian basis functions
 * (upper) to transform to the spherical basis functions (lower).
 *
 * The first remaining task is to specify a lexical ordering of the spherical
 * basis functions. In LibGaussian we adopt the convention (c is for cos, s is
 * for sin):
 *  l0,l1c,l1s,l2c,l2s,..., 
 * which is equivalent to the +/- convention sometimes seen,
 *  l,0,l,+1,l,-1,l,+2,l,-2,...
 *
 * The second remaining task is to specify the coefficient matrix C_lm^lmn in a
 * sparse manner, which is accomplished below in the
 * cartesian_ind/spherical_ind/cartesian_coef methods (and pural variants
 * thereof).
 *
 * For reference, these coefficients are derived by recurrence relations as may
 * be found in many places in the literature. In particular, we have
 * implemented the equations directly from Equations 6.4.70-6.4.73 in Molecular
 * Electronic-Structure Theory by Helgaker, Jorgensen, and Olsen (the Purple
 * Book):
 *
 *  S_0,+0 = 1 
 *  S_l+1,+l+1 = sqrt{2^{\delta_{l0}} \frac{2l+1}{2l+2}} (x S_l,l - (1 - \delta_{l0}) y S_l,-l) 
 *  S_l+1,-l+1 = sqrt{2^{\delta_{l0}} \frac{2l+1}{2l+2}} (y S_l,l + (1 - \delta_{l0}) x S_l,-l) 
 *  S_l+1,m = \frac{(2l+1) z S_l,m - sqrt{(l+m)(l-m)} r^2 S_l-1,m}{\sqrt{(l+m+1)(l+m-1)}} 
 *
 * The explicit solid harmonics up through D-type shells are specified below.
 * The corresponding expressions for F- and G-type shells are explicitly
 * presented in Table 6.3 of the Purple Book.
 *
 *  S_00  = 1
 *  S_10  = z
 *  S_11c = x
 *  S_11c = y
 *  S_20  = 1/2 (3 zz - rr)
 *  S_21c = \sqrt{3} xz
 *  S_21s = \sqrt{3} yz
 *  S_22c = 1/2 \sqrt{3} (xx - yy) 
 *  S_22s = \sqrt{3} xy 
 *
 * - Rob Parrish, 16 February, 2015
 **/
class SAngularMomentum {

public:

    // => Constructors <= //

    /// Main constructor, builds information for a shell with L = am
    SAngularMomentum(int am);
    /// Default constructor, no initialization
    SAngularMomentum() {}  

    /// Return a vector of SAngularMomentum in [0,Lmax] (Lmax + 1 entries)
    static std::vector<SAngularMomentum> build(int Lmax);

    // => General Accessors <= //

    /// Angular momentum of this type of shell
    int am() const { return am_; }
    /// Number of cartesian functions in this type of shell
    size_t ncartesian() const { return (am_ + 1L) * (am_ + 2L) / 2L; }
    /// Number of spherical functions in this type of shell
    size_t nspherical() const { return 2L * am_ + 1L; }
    
    // => Cartesian Shell Information <= //

    /// Powers of x in the cartesian version of this shell, ind in [0,ncartesian())
    int l(size_t ind) const { return ls_[ind]; }
    /// Powers of y in the cartesian version of this shell, ind in [0,ncartesian())
    int m(size_t ind) const { return ms_[ind]; }
    /// Powers of z in the cartesian version of this shell, ind in [0,ncartesian())
    int n(size_t ind) const { return ns_[ind]; }
    /// Powers of x in the cartesian version of this shell, length ncartesian()
    const std::vector<int>& ls() const { return ls_; }
    /// Powers of y in the cartesian version of this shell, length ncartesian()
    const std::vector<int>& ms() const { return ms_; }
    /// Powers of z in the cartesian version of this shell, length ncartesian()
    const std::vector<int>& ns() const { return ns_; }

    // => Spherical Shell Information <= //

    /// Total number of cartesian to spherical transformation coefficients
    size_t ncoef() const { return cartesian_inds_.size(); }
    /// Indices of cartesian functions in transformation, ind in [0, ncoef())
    int cartesian_ind(size_t ind) const { return cartesian_inds_[ind]; }
    /// Indices of spherical functions in transformation, ind in [0, ncoef())
    int spherical_ind(size_t ind) const { return spherical_inds_[ind]; }
    /// Coefficients of cartesian functions in transformation, ind in [0, ncoef())
    double cartesian_coef(size_t ind) const { return cartesian_coefs_[ind]; }
    /// Indices of cartesian functions in transformation, length ncoef()
    const std::vector<int>& cartesian_inds() const { return cartesian_inds_; }
    /// Indices of spherical functions in transformation, length ncoef()
    const std::vector<int>& spherical_inds() const { return spherical_inds_; }
    /// Coefficients of cartesian functions in transformation, length ncoef()
    const std::vector<double>& cartesian_coefs() const { return cartesian_coefs_; }

private:

    int am_;

    std::vector<int> ls_;
    std::vector<int> ms_;
    std::vector<int> ns_;

    std::vector<int> cartesian_inds_;
    std::vector<int> spherical_inds_;
    std::vector<double> cartesian_coefs_;

    void build_cartesian();
    void build_spherical();

};

} // namespace libgaussian

#endif
