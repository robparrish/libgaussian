#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <string.h>
#include <cassert>
#include "fjt.h"
#include "constants.h"

namespace lightspeed {

const double oon[] = {0.0, 1.0, 1.0 / 2.0, 1.0 / 3.0, 1.0 / 4.0, 1.0 / 5.0, 1.0 / 6.0, 1.0 / 7.0, 1.0 / 8.0, 1.0 / 9.0, 1.0 / 10.0, 1.0 / 11.0};

#define SOFT_ZERO 1e-6

#ifndef M_SQRT_PI
#define M_SQRT_PI    1.772453850905516027298167483341    /* sqrt(pi) */
#endif

#ifndef M_SQRT_PI_2
#define M_SQRT_PI_2 1.2533141373155002512078826424055   // sqrt(Pi/2)
#endif

namespace {

double ** block_matrix(unsigned long int n, unsigned long int m, bool memlock=true)
{
    double **A=NULL;
    double *B=NULL;
    unsigned long int i;

    if(!m || !n) return(static_cast<double **>(0));

    A = new double*[n];
    if (A==NULL) {
        printf("block_matrix: trouble allocating memory \n");
        printf("n = %ld\n",n);
        abort();
    }

    B = new double[n*m];
    if (B == NULL) {
        printf("block_matrix: trouble allocating memory \n");
        printf("m = %ld\n",m);
        abort();
    }
    memset(static_cast<void*>(B), 0, m*n*sizeof(double));

    for (i = 0; i < n; i++) {
        A[i] = &(B[i*m]);
    }

#ifdef _POSIX_MEMLOCK
    if (memlock) {

        char* addr = (char*) B;
        unsigned long size = m*n*(unsigned long)sizeof(double);
        unsigned long page_offset, page_size;

        page_size = sysconf(_SC_PAGESIZE);
        page_offset = (unsigned long) addr % page_size;

        addr -= page_offset;  /* Adjust addr to page boundary */
        size += page_offset;  /* Adjust size with page_offset */

        if ( mlock(addr, size) ) {  /* Lock the memory */
            outfile->Printf("block_matrix: trouble locking memory \n");
            fflush(stderr);
            exit(PSI_RETURN_FAILURE);
        }

        addr = (char*) A;
        size = n*(unsigned long)sizeof(double*);

        page_offset = (unsigned long) addr % page_size;

        addr -= page_offset;  /* Adjust addr to page boundary */
        size += page_offset;  /* Adjust size with page_offset */

        if ( mlock(addr, size) ) {  /* Lock the memory */
            outfile->Printf("block_matrix: trouble locking memory \n");
            fflush(stderr);
            exit(PSI_RETURN_FAILURE);
        }
    }
#endif

    return(A);
}


void free_block(double **array)
{
    if(array == NULL) return;
    delete [] array[0];
    delete [] array;
}

} // anonmyous namespace

namespace fundamentals {

namespace parameters {

CorrelationFactor::CorrelationFactor(const std::vector<double> &coeff,
                                     const std::vector<double> &exponent)
        : coeff_(coeff), exponent_(exponent)
{
    assert(coeff.size() == exponent.size());
}

void CorrelationFactor::set_params(const std::vector<double>& coeff, const std::vector<double>& exponent)
{
    assert(coeff.size() == exponent.size());

    coeff_ = coeff;
    exponent_ = exponent;
}

FittedSlaterCorrelationFactor::FittedSlaterCorrelationFactor(const double& exponent)
{
    coeff_.resize(6);
    exponent_.resize(6);

    slater_exponent_ = exponent;

    // The fitting coefficients
    coeff_[0] = -0.3144;
    coeff_[1] = -0.3037;
    coeff_[2] = -0.1681;
    coeff_[3] = -0.09811;
    coeff_[4] = -0.06024;
    coeff_[5] = -0.03726;

    double expsq = exponent * exponent;

    exponent_[0] = 0.2209 * expsq;
    exponent_[1] = 1.004 * expsq;
    exponent_[2] = 3.622 * expsq;
    exponent_[3] = 12.16 * expsq;
    exponent_[4] = 45.87 * expsq;
    exponent_[5] = 254.4 * expsq;
}

} // namespace parameters

namespace base {

Fjt::Fjt(size_t max)
    : rho_(0.0), values_(max)
{
}
Fjt::~Fjt()
{
}

}

#define TAYLOR_INTERPOLATION_ORDER 6
#define TAYLOR_INTERPOLATION_AND_RECURSION  0

double Taylor_Fjt::relative_zero_(1e-6);

/*------------------------------------------------------
  Initialize Taylor_Fm_Eval object (computes incomplete
  gamma function via Taylor interpolation)
 ------------------------------------------------------*/
Taylor_Fjt::Taylor_Fjt(size_t mmax, double accuracy)
    : Fjt(mmax + 1),
      cutoff_(accuracy), interp_order_(TAYLOR_INTERPOLATION_ORDER)
{
    const double sqrt_pi = M_SQRT_PI;

    /*---------------------------------------
    We are doing Taylor interpolation with
    n=TAYLOR_ORDER terms here:
    error <= delT^n/(n+1)!
   ---------------------------------------*/
    delT_ = 2.0 * std::pow(cutoff_ * constants::fac__[interp_order_ + 1],
                           1.0 / interp_order_);
    oodelT_ = 1.0 / delT_;
    max_m_ = mmax + interp_order_ - 1;

    T_crit_ = new double[max_m_ + 1];   /*--- m=0 is included! ---*/
    max_T_ = 0;
    /*--- Figure out T_crit for each m and put into the T_crit ---*/
    for (int m = max_m_; m >= 0; --m) {
        /*------------------------------------------
      Damped Newton-Raphson method to solve
      T^{m-0.5}*exp(-T) = epsilon*Gamma(m+0.5)
      The solution is the max T for which to do
      the interpolation
     ------------------------------------------*/
        double T = -log(cutoff_);
        const double egamma = cutoff_ * sqrt_pi * constants::df__[2 * m] / std::pow(2.0, m);
        double T_new = T;
        double func;
        do {
            const double damping_factor = 0.2;
            T = T_new;
            /* f(T) = the difference between LHS and RHS of the equation above */
            func = std::pow(T, m - 0.5) * std::exp(-T) - egamma;
            const double dfuncdT = ((m - 0.5) * std::pow(T, m - 1.5) - std::pow(T, m - 0.5)) * std::exp(-T);
            /* f(T) has 2 roots and has a maximum in between. If f'(T) > 0 we are to the left of the hump. Make a big step to the right. */
            if (dfuncdT > 0.0) {
                T_new *= 2.0;
            }
            else {
                /* damp the step */
                double deltaT = -func / dfuncdT;
                const double sign_deltaT = (deltaT > 0.0) ? 1.0 : -1.0;
                const double max_deltaT = damping_factor * T;
                if (std::fabs(deltaT) > max_deltaT)
                    deltaT = sign_deltaT * max_deltaT;
                T_new = T + deltaT;
            }
            if (T_new <= 0.0) {
                T_new = T / 2.0;
            }
        } while (std::fabs(func / egamma) >= SOFT_ZERO);
        T_crit_[m] = T_new;
        const int T_idx = (int) std::floor(T_new / delT_);
        max_T_ = std::max(max_T_, T_idx);
    }

    // allocate the grid (see the comments below)
    {
        const size_t nrow = max_T_ + 1;
        const size_t ncol = max_m_ + 1;
        grid_ = block_matrix(nrow, ncol);
        //grid_ = new double*[nrow];
        //grid_[0] = new double[nrow*ncol];
        //for(int r=1; r<nrow; ++r)
        //    grid_[r] = grid_[r-1] + ncol;
    }

    /*-------------------------------------------------------
    Tabulate the gamma function from t=delT to T_crit[m]:
    1) include T=0 though the table is empty for T=0 since
       Fm(0) is simple to compute
    2) modified MacLaurin series converges fastest for
       the largest m -> use it to compute Fmmax(T)
       see JPC 94, 5564 (1990).
    3) then either use the series to compute the rest
       of the row or maybe use downward recursion
   -------------------------------------------------------*/
    /*--- do the mmax first ---*/
    const double cutoff_o_10 = 0.1 * cutoff_;
    for (int m = 0; m <= max_m_; ++m) {
        for (int T_idx = max_T_;
             T_idx >= 0;
             --T_idx) {
            const double T = T_idx * delT_;
            double denom = (m + 0.5);
            double term = 0.5 * std::exp(-T) / denom;
            double sum = term;
            //            double rel_error;
            double epsilon;
            do {
                denom += 1.0;
                term *= T / denom;
                sum += term;
                //                rel_error = term/sum;
                // stop if adding a term smaller or equal to cutoff_/10 and smaller than relative_zero * sum
                // When sum is small in absolute value, the second threshold is more important
                epsilon = std::min(cutoff_o_10, sum * relative_zero_);
            } while (term > epsilon);
            //            } while (term > epsilon || term > sum*relative_zero_);

            grid_[T_idx][m] = sum;
        }
    }
}

Taylor_Fjt::~Taylor_Fjt()
{
    delete[] T_crit_;
    T_crit_ = 0;
    free_block(grid_);
    grid_ = NULL;
}

/* Using the tabulated incomplete gamma function in gtable, compute
 * the incomplete gamma function for a particular wval for all 0<=j<=J.
 * The result is placed in the global intermediate int_fjttable.
 */
void Taylor_Fjt::compute(size_t l, double T)
{
    const double two_T = 2.0 * T;

    // since Tcrit grows with l, this condition only needs to be determined once
    const bool T_gt_Tcrit = T > T_crit_[l];
    // start recursion at j=jrecur
    const size_t jrecur = TAYLOR_INTERPOLATION_AND_RECURSION ? l : 0;
    /*-------------------------------------
     Compute Fj(T) from l down to jrecur
   -------------------------------------*/
    if (T_gt_Tcrit) {
#define UPWARD_RECURSION 1
#define AVOID_POW 1 // Pow is only used in the downwards recursion case
#if UPWARD_RECURSION
#if TAYLOR_INTERPOLATION_AND_RECURSION
        #error upward recursion cannot be used with taylor interpolation
    #endif
        double X = 1.0 / two_T;
        double dffac = 1.0;
        double jfac = 1.0; // (j!! X^-j)
        double Fj = M_SQRT_PI_2 * sqrt(X); // Start with F0; this is why interpolation can't be used
        for (int j = 0; j < l; ++j) {
            /*--- Asymptotic formula, c.f. IJQC 40 745 (1991) ---*/
            values_[j] = jfac * Fj;
            jfac *= dffac * X;
            dffac += 2.0;
        }
        values_[l] = jfac * Fj;
#else
    #if AVOID_POW
        double X = 1.0/two_T;
        double pow_two_T_to_minusjp05 = X;
        for(int i = 0; i < l; ++i)
            pow_two_T_to_minusjp05 *= X*X;
        pow_two_T_to_minusjp05 = sqrt(pow_two_T_to_minusjp05);
    #else
        double pow_two_T_to_minusjp05 = std::pow(two_T,-l-0.5);
    #endif
    for(int j=l; j>=jrecur; --j) {
        /*--- Asymptotic formula ---*/
        F_[j] = df[2*j] * M_SQRT_PI_2 * pow_two_T_to_minusjp05;
        pow_two_T_to_minusjp05 *= two_T;
    }
#endif
    }
    else {
        const int T_ind = (int) std::floor(0.5 + T * oodelT_);
        const double h = T_ind * delT_ - T;
        const double *F_row = grid_[T_ind] + l;

        for (int j = l; j >= jrecur; --j, --F_row) {

            /*--- Taylor interpolation ---*/
            values_[j] = F_row[0]
#if TAYLOR_INTERPOLATION_ORDER > 0
                    + h * (F_row[1]
#endif
#if TAYLOR_INTERPOLATION_ORDER > 1
                    + oon[2] * h * (F_row[2]
#endif
#if TAYLOR_INTERPOLATION_ORDER > 2
                    + oon[3] * h * (F_row[3]
#endif
#if TAYLOR_INTERPOLATION_ORDER > 3
                    + oon[4] * h * (F_row[4]
#endif
#if TAYLOR_INTERPOLATION_ORDER > 4
                    + oon[5] * h * (F_row[5]
#endif
#if TAYLOR_INTERPOLATION_ORDER > 5
                    + oon[6] * h * (F_row[6]
#endif
#if TAYLOR_INTERPOLATION_ORDER > 6
                                                                                      +oon[7]*h*(F_row[7]
                                                                                     #endif
#if TAYLOR_INTERPOLATION_ORDER > 7
                                                                                                 +oon[8]*h*(F_row[8])
                                                                                     #endif
#if TAYLOR_INTERPOLATION_ORDER > 6
                                                                                                 )
                                                                          #endif
#if TAYLOR_INTERPOLATION_ORDER > 5
            )
#endif
#if TAYLOR_INTERPOLATION_ORDER > 4
            )
#endif
#if TAYLOR_INTERPOLATION_ORDER > 3
            )
#endif
#if TAYLOR_INTERPOLATION_ORDER > 2
            )
#endif
#if TAYLOR_INTERPOLATION_ORDER > 1
            )
#endif
#if TAYLOR_INTERPOLATION_ORDER > 0
            )
#endif
                    ;
        } // interpolation for F_j(T), jrecur<=j<=l
    } // if T < T_crit

    /*------------------------------------
    And then do downward recursion in j
   ------------------------------------*/
#if TAYLOR_INTERPOLATION_AND_RECURSION
    if (l > 0 && jrecur > 0) {
        double F_jp1 = F_[jrecur];
        const double exp_jT = std::exp(-T);
        for(int j=jrecur-1; j>=0; --j) {
            const double F_j = (exp_jT + two_T*F_jp1)*oo2np1[j];
            F_[j] = F_j;
            F_jp1 = F_j;
        }
    }
#endif
}

/////////////////////////////////////////////////////////////////////////////

/* Tablesize should always be at least 121. */
#define TABLESIZE 121

/* Tabulate the incomplete gamma function and put in gtable. */
/*
 *     For J = JMAX a power series expansion is used, see for
 *     example Eq.(39) given by V. Saunders in "Computational
 *     Techniques in Quantum Chemistry and Molecular Physics",
 *     Reidel 1975.  For J < JMAX the values are calculated
 *     using downward recursion in J.
 */
FJT::FJT(size_t max)
    : Fjt(max + 1)
{
    int i, j;
    double denom, d2jmax1, r2jmax1, wval, d2wval, sum, term, rexpw;

    maxj = max;

    /* Allocate storage for gtable and int_fjttable. */
    gtable = new double *[ngtable()];
    for (i = 0; i < ngtable(); i++) {
        gtable[i] = new double[TABLESIZE];
    }

    /* Tabulate the gamma function for t(=wval)=0.0. */
    denom = 1.0;
    for (i = 0; i < ngtable(); i++) {
        gtable[i][0] = 1.0 / denom;
        denom += 2.0;
    }

    /* Tabulate the gamma function from t(=wval)=0.1, to 12.0. */
    d2jmax1 = 2.0 * (ngtable() - 1) + 1.0;
    r2jmax1 = 1.0 / d2jmax1;
    for (i = 1; i < TABLESIZE; i++) {
        wval = 0.1 * i;
        d2wval = 2.0 * wval;
        term = r2jmax1;
        sum = term;
        denom = d2jmax1;
        for (j = 2; j <= 200; j++) {
            denom = denom + 2.0;
            term = term * d2wval / denom;
            sum = sum + term;
            if (term <= 1.0e-15) break;
        }
        rexpw = exp(-wval);

        /* Fill in the values for the highest j gtable entries (top row). */
        gtable[ngtable() - 1][i] = rexpw * sum;

        /* Work down the table filling in the rest of the column. */
        denom = d2jmax1;
        for (j = ngtable() - 2; j >= 0; j--) {
            denom = denom - 2.0;
            gtable[j][i] = (gtable[j + 1][i] * d2wval + rexpw) / denom;
        }
    }

    /* Form some denominators, so divisions can be eliminated below. */
    denomarray = new double[max + 1];
    denomarray[0] = 0.0;
    for (i = 1; i <= max; i++) {
        denomarray[i] = 1.0 / (2 * i - 1);
    }

    wval_infinity = 2 * max + 37.0;
    itable_infinity = (size_t)(10 * wval_infinity);

}

FJT::~FJT()
{
    for (int i = 0; i < maxj + 7; i++) {
        delete[] gtable[i];
    }
    delete[] gtable;
    delete[] denomarray;
}

/* Using the tabulated incomplete gamma function in gtable, compute
 * the incomplete gamma function for a particular wval for all 0<=j<=J.
 * The result is placed in the global intermediate int_fjttable.
 */
void FJT::compute(size_t J, double wval)
{
    const double sqrpih = 0.886226925452758;
    const double coef2 = 0.5000000000000000;
    const double coef3 = -0.1666666666666667;
    const double coef4 = 0.0416666666666667;
    const double coef5 = -0.0083333333333333;
    const double coef6 = 0.0013888888888889;
    const double gfac30 = 0.4999489092;
    const double gfac31 = -0.2473631686;
    const double gfac32 = 0.321180909;
    const double gfac33 = -0.3811559346;
    const double gfac20 = 0.4998436875;
    const double gfac21 = -0.24249438;
    const double gfac22 = 0.24642845;
    const double gfac10 = 0.499093162;
    const double gfac11 = -0.2152832;
    const double gfac00 = -0.490;

    double wdif, d2wal, rexpw, /* denom, */ gval, factor, rwval, term;
    int i, itable, irange;

    if (J > maxj) {
        printf("the int_fjt routine has been incorrectly used\n");
        printf("J = %d but maxj = %zu\n", J, maxj);
        abort();
    }

    /* Compute an index into the table. */
    /* The test is needed to avoid floating point exceptions for
   * large values of wval. */
    if (wval > wval_infinity) {
        itable = itable_infinity;
    }
    else {
        itable = (int) (10.0 * wval);
    }

    /* If itable is small enough use the table to compute int_fjttable. */
    if (itable < TABLESIZE) {

        wdif = wval - 0.1 * itable;

        /* Compute fjt for J. */
        values_[J] = (((((coef6 * gtable[J + 6][itable] * wdif
                + coef5 * gtable[J + 5][itable]) * wdif
                + coef4 * gtable[J + 4][itable]) * wdif
                + coef3 * gtable[J + 3][itable]) * wdif
                + coef2 * gtable[J + 2][itable]) * wdif
                - gtable[J + 1][itable]) * wdif
                + gtable[J][itable];

        /* Compute the rest of the fjt. */
        d2wal = 2.0 * wval;
        rexpw = exp(-wval);
        /* denom = 2*J + 1; */
        for (i = J; i > 0; i--) {
            /* denom = denom - 2.0; */
            values_[i - 1] = (d2wal * values_[i] + rexpw) * denomarray[i];
        }
    }
        /* If wval <= 2*J + 36.0, use the following formula. */
    else if (itable <= 20 * J + 360) {
        rwval = 1.0 / wval;
        rexpw = exp(-wval);

        /* Subdivide wval into 6 ranges. */
        irange = itable / 30 - 3;
        if (irange == 1) {
            gval = gfac30 + rwval * (gfac31 + rwval * (gfac32 + rwval * gfac33));
            values_[0] = sqrpih * sqrt(rwval) - rexpw * gval * rwval;
        }
        else if (irange == 2) {
            gval = gfac20 + rwval * (gfac21 + rwval * gfac22);
            values_[0] = sqrpih * sqrt(rwval) - rexpw * gval * rwval;
        }
        else if (irange == 3 || irange == 4) {
            gval = gfac10 + rwval * gfac11;
            values_[0] = sqrpih * sqrt(rwval) - rexpw * gval * rwval;
        }
        else if (irange == 5 || irange == 6) {
            gval = gfac00;
            values_[0] = sqrpih * sqrt(rwval) - rexpw * gval * rwval;
        }
        else {
            values_[0] = sqrpih * sqrt(rwval);
        }

        /* Compute the rest of the int_fjttable from int_fjttable[0]. */
        factor = 0.5 * rwval;
        term = factor * rexpw;
        for (i = 1; i <= J; i++) {
            values_[i] = factor * values_[i - 1] - term;
            factor = rwval + factor;
        }
    }
        /* For large values of wval use this algorithm: */
    else {
        rwval = 1.0 / wval;
        values_[0] = sqrpih * sqrt(rwval);
        factor = 0.5 * rwval;
        for (i = 1; i <= J; i++) {
            values_[i] = factor * values_[i - 1];
            factor = rwval + factor;
        }
    }
}

GaussianFundamental::GaussianFundamental(const parameters::CorrelationFactor &cf, size_t max)
    : Fjt(max+1), cf_(cf)
{
}

F12::F12(const parameters::CorrelationFactor &cf, size_t max)
    : GaussianFundamental(cf, max)
{
}

void F12::compute(size_t J, double T)
{
    const double *e = cf_.exponent().data();
    const double *c = cf_.coeff().data();
    size_t nparam = cf_.exponent().size();

    // zero the values array
    for (int n=0; n<=J; ++n)
        values_[n] = 0.0;

    double pfac, expterm, rhotilde, omega;
    double eri_correct = rho_ / 2 / M_PI;
    for (int i=0; i<nparam; ++i) {
        omega = e[i];
        rhotilde = omega / (rho_ + omega);
        pfac = c[i] * pow(M_PI/(rho_ + omega), 1.5) * eri_correct;
        expterm = exp(-rhotilde*T)*pfac;
        for (int n=0; n<=J; ++n) {
            values_[n] += expterm;
            expterm *= rhotilde;
        }
    }
}

F12Scaled::F12Scaled(const parameters::CorrelationFactor &cf, size_t max)
    : GaussianFundamental(cf, max)
{
}

void F12Scaled::compute(size_t J, double T)
{
    const double *e = cf_.exponent().data();
    const double *c = cf_.coeff().data();
    size_t nparam = cf_.exponent().size();

    // zero the values array
    for (int n=0; n<=J; ++n)
        values_[n] = 0.0;

    double pfac, expterm, rhotilde, omega;
    double eri_correct = rho_ / 2 / M_PI;
    eri_correct /= cf_.slater_exponent();
    for (int i=0; i<nparam; ++i) {
        omega = e[i];
        rhotilde = omega / (rho_ + omega);
        pfac = c[i] * pow(M_PI/(rho_ + omega), 1.5) * eri_correct;
        expterm = exp(-rhotilde*T)*pfac;
        for (int n=0; n<=J; ++n) {
            values_[n] += expterm;
            expterm *= rhotilde;
        }
    }
}

F12Squared::F12Squared(const parameters::CorrelationFactor &cf, size_t max)
    : GaussianFundamental(cf, max)
{
}

void F12Squared::compute(size_t J, double T)
{
    const double *e = cf_.exponent().data();
    const double *c = cf_.coeff().data();
    size_t nparam = cf_.exponent().size();

    double pfac, expterm, rhotilde, omega;
    double eri_correct = rho_ / 2 / M_PI;

    // zero the values
    for (int n=0; n<=J; ++n)
        values_[n] = 0.0;

    for (int i=0; i<nparam; ++i) {
        for (int j=0; j<nparam; ++j) {
            omega = e[i] + e[j];
            rhotilde = omega / (rho_ + omega);
            pfac = c[i] * c[j] * pow(M_PI/(rho_+omega), 1.5) * eri_correct;
            expterm = exp(-rhotilde * T) * pfac;
            for (int n=0; n<=J; ++n) {
                values_[n] += expterm;
                expterm *= rhotilde;
            }
        }
    }
}

F12G12::F12G12(const parameters::CorrelationFactor &cf, size_t max)
    : GaussianFundamental(cf, max), Fm_(max)
{
}

void F12G12::compute(size_t J, double T)
{
    const std::vector<double>& Fvals = Fm_.values();

    const double *e = cf_.exponent().data();
    const double *c = cf_.coeff().data();
    size_t nparam = cf_.exponent().size();

    double pfac, expterm, rhotilde, omega, rhohat;
    double boysterm, rhotilde_term, rhohat_term;
    double eri_correct = rho_ / 2 / M_PI;
    double binom_term;

    // Zero the values
    for (int n=0; n<=J; ++n)
        values_[n] = 0.0;

    for (int i=0; i<nparam; ++i) {
        omega = e[i];
        rhotilde = omega / (rho_ + omega);
        rhohat = rho_ / (rho_ + omega);
        expterm = exp(-rhotilde * T);
        pfac = 2*M_PI / (rho_ + omega) * c[i] * expterm * eri_correct;
        Fm_.compute(J, rhohat * T);
        for (int n=0; n<=J; ++n) {
            boysterm = 0.0;
            rhotilde_term = pow(rhotilde, n);
            rhohat_term = 1.0;
            for (int m=0; m<=n; ++m) {
                binom_term = constants::bc__[n][m];
                boysterm += binom_term * rhotilde_term * rhohat_term * Fvals[m];

                rhotilde_term /= rhotilde;
                rhohat_term *= rhohat;
            }
            values_[n] += pfac * boysterm;
        }
    }
}

F12DoubleCommutator::F12DoubleCommutator(const parameters::CorrelationFactor &cf, size_t max)
    : GaussianFundamental(cf, max)
{
}

void F12DoubleCommutator::compute(size_t J, double T)
{
    const double *e = cf_.exponent().data();
    const double *c = cf_.coeff().data();
    size_t nparam = cf_.exponent().size();

    double pfac, expterm, rhotilde, omega, sqrt_term, rhohat;
    double eri_correct = rho_ / 2 / M_PI;
    double term1, term2;

    // Zero the values
    for (int n=0; n<=J; ++n)
        values_[n] = 0.0;

    for (int i=0; i<nparam; ++i) {
        for (int j=0; j<nparam; ++j) {
            omega = e[i] + e[j];
            rhotilde = omega / (rho_ + omega);
            rhohat = rho_ / (rho_ + omega);
            expterm = exp(-rhotilde * T);
            sqrt_term = sqrt(M_PI*M_PI*M_PI / pow(rho_ + omega, 5.0));
            pfac = 4.0*c[i] * c[j] * e[i] * e[j] * sqrt_term * eri_correct * expterm;

            term1 = 1.5*rhotilde + rhotilde*rhohat*T;
            term2 = 1.0/rhotilde*pfac;
            for (int n=0; n<=J; ++n) {
                values_[n] += term1 * term2;
                term1 -= rhohat;
                term2 *= rhotilde;
            }
        }
    }
}

ErfFundamental::ErfFundamental(double omega, size_t max)
    : Fjt(max+1), omega_(omega), boys_(max)
{
}

void ErfFundamental::compute(size_t J, double T)
{
    const std::vector<double>& Fvals = boys_.values();
    boys_.compute(J, T);

    for (int n=0; n<=J; ++n)
        values_[n] = 0.0;

    // build the erf constants
    double omegasq = omega_ * omega_;
    double T_prefac = omegasq / (omegasq + rho_);
    double F_prefac = sqrt(T_prefac);
    double erf_T = T_prefac * T;

    boys_.compute(J, erf_T);
    for (int n=0; n<=J; ++n) {
        values_[n] += Fvals[n] * F_prefac;
        F_prefac *= T_prefac;
    }
}

ErfComplementFundamental::ErfComplementFundamental(double omega, size_t max)
    : Fjt(max+1), omega_(omega), boys_(max)
{
}

void ErfComplementFundamental::compute(size_t J, double T)
{
    const std::vector<double>& Fvals = boys_.values();
    boys_.compute(J, T);

    for (int n=0; n<=J; ++n)
        values_[n] = Fvals[n];

    // build the erf constants
    double omegasq = omega_ * omega_;
    double T_prefac = omegasq / (omegasq + rho_);
    double F_prefac = sqrt(T_prefac);
    double erf_T = T_prefac * T;

    boys_.compute(J, erf_T);
    for (int n=0; n<=J; ++n) {
        values_[n] -= Fvals[n] * F_prefac;
        F_prefac *= T_prefac;
    }
}

} // namespace fundamentals
} // namespace lightspeed
