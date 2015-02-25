#ifndef _chemistry_qc_basis_fjt_h
#define _chemistry_qc_basis_fjt_h

namespace lightspeed {

class CorrelationFactor;

/// Evaluates the Boys function F_j(T)
class Fjt {
public:
    Fjt();
    virtual ~Fjt();
    /** Computed F_j(T) for every 0 <= j <= J (total of J+1 doubles).
        The user may read/write these values.
        The values will be overwritten with the next call to this functions.
        The pointer will be invalidated after the call to ~Fjt. */
    virtual double *values(int J, double T) =0;
    virtual void set_rho(double /*rho*/) { }
};

#define TAYLOR_INTERPOLATION_ORDER 6
#define TAYLOR_INTERPOLATION_AND_RECURSION 0  // compute F_lmax(T) and then iterate down to F_0(T)? Else use interpolation only
/// Uses Taylor interpolation of up to 8-th order to compute the Boys function
class Taylor_Fjt : public Fjt {
    static double relative_zero_;
public:
    static const int max_interp_order = 8;

    Taylor_Fjt(unsigned int jmax, double accuracy);
    virtual ~Taylor_Fjt();
    /// Implements Fjt::values()
    double *values(int J, double T);
private:
    double **grid_;            /* Table of "exact" Fm(T) values. Row index corresponds to
                                  values of T (max_T+1 rows), column index to values
                                  of m (max_m+1 columns) */
    double delT_;              /* The step size for T, depends on cutoff */
    double oodelT_;            /* 1.0 / delT_, see above */
    double cutoff_;            /* Tolerance cutoff used in all computations of Fm(T) */
    int interp_order_;         /* Order of (Taylor) interpolation */
    int max_m_;                /* Maximum value of m in the table, depends on cutoff
                                  and the number of terms in Taylor interpolation */
    int max_T_;                /* Maximum index of T in the table, depends on cutoff
                                  and m */
    double *T_crit_;           /* Maximum T for each row, depends on cutoff;
                                  for a given m and T_idx <= max_T_idx[m] use Taylor interpolation,
                                  for a given m and T_idx > max_T_idx[m] use the asymptotic formula */
    double *F_;                /* Here computed values of Fj(T) are stored */
};

/// "Old" intv3 code from Curt
/// Computes F_j(T) using 6-th order Taylor interpolation
class FJT: public Fjt {
private:
    double **gtable;

    int maxj;
    double *denomarray;
    double wval_infinity;
    int itable_infinity;

    double *int_fjttable;

    int ngtable() const { return maxj + 7; }
public:
    FJT(int n);
    virtual ~FJT();
    /// implementation of Fjt::values()
    double *values(int J, double T);
};

} // end of namespace sc

#endif // header guard

// Local Variables:
// mode: c++
// c-file-style: "CLJ"
// End:
