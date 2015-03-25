#ifndef FJT_H
#define FJT_H

#include <vector>

namespace lightspeed {

namespace fundamentals {

namespace parameters {

class CorrelationFactor
{
protected:
    std::vector<double> coeff_;
    std::vector<double> exponent_;

public:

    CorrelationFactor() = default;

    CorrelationFactor(const std::vector<double> &coeff,
                      const std::vector<double> &exponent);

    virtual double slater_exponent() const
    {
        return 1.0;
    }

    void set_params(const std::vector<double>& coeff,
                    const std::vector<double>& exponent);

    const std::vector<double> &exponent() const
    {
        return exponent_;
    }
    const std::vector<double> &coeff() const
    {
        return coeff_;
    }
};

class FittedSlaterCorrelationFactor final : public CorrelationFactor
{
private:
    double slater_exponent_;

public:

    double slater_exponent() const
    {
        return slater_exponent_;
    }

    FittedSlaterCorrelationFactor(const double& exponent);

//    double exponent()
//    {
//        return slater_exponent_;
//    }
};

} // namespace parameters

namespace base {

/// This is the base class for the supported fundamentals.
class Fjt
{
protected:
    double rho_;
    std::vector<double> values_;

public:
    Fjt(size_t max);

    virtual ~Fjt();

    const std::vector<double> &values() const
    {
        return values_;
    }

    /** Computed F_j(T) for every 0 <= j <= J (total of J+1 doubles).
        The user may read/write these values.
        The values will be overwritten with the next call to this functions.
        The pointer will be invalidated after the call to ~Fjt. */
    virtual void compute(size_t J, double T) = 0;

    void set_rho(double rho)
    {
        rho_ = rho;
    }

    virtual void set_omega(double) {}
};

}

/// Uses Taylor interpolation of up to 8-th order to compute the Boys function
class Taylor_Fjt : public base::Fjt
{
    static double relative_zero_;
public:
    static const int max_interp_order = 8;

    Taylor_Fjt(size_t jmax, double accuracy);

    virtual ~Taylor_Fjt();

    /// Implements Fjt::values()
    void compute(size_t J, double T);

private:
    double **grid_;
    /* Table of "exact" Fm(T) values. Row index corresponds to
                                  values of T (max_T+1 rows), column index to values
                                  of m (max_m+1 columns) */
    double delT_;
    /* The step size for T, depends on cutoff */
    double oodelT_;
    /* 1.0 / delT_, see above */
    double cutoff_;
    /* Tolerance cutoff used in all computations of Fm(T) */
    int interp_order_;
    /* Order of (Taylor) interpolation */
    int max_m_;
    /* Maximum value of m in the table, depends on cutoff
                                  and the number of terms in Taylor interpolation */
    int max_T_;
    /* Maximum index of T in the table, depends on cutoff
                                  and m */
    double *T_crit_;
    /* Maximum T for each row, depends on cutoff;
                                  for a given m and T_idx <= max_T_idx[m] use Taylor interpolation,
                                  for a given m and T_idx > max_T_idx[m] use the asymptotic formula */
//    double *F_;                /* Here computed values of Fj(T) are stored */
};

/// "Old" intv3 code from Curt
/// Computes F_j(T) using 6-th order Taylor interpolation
class FJT : public base::Fjt
{
private:
    double **gtable;

    int maxj;
    double *denomarray;
    double wval_infinity;
    int itable_infinity;

//    double *int_fjttable;

    int ngtable() const
    {
        return maxj + 7;
    }
public:
    FJT(size_t n);

    virtual ~FJT();

    void compute(size_t J, double T);
};

class GaussianFundamental : public base::Fjt
{
protected:
    const parameters::CorrelationFactor &cf_;

public:
    GaussianFundamental(const parameters::CorrelationFactor& cf, size_t max);

    virtual void compute(size_t J, double T) = 0;
};

/**
 *  Solves \scp -\gamma r_{12}
 */
class F12 : public GaussianFundamental
{
public:
    F12(const parameters::CorrelationFactor& cf, size_t max);

    void compute(size_t J, double T);
};

/**
 *  Solves \frac{\exp -\gamma r_{12}}{\gamma}.
 */
class F12Scaled : public GaussianFundamental
{
public:
    F12Scaled(const parameters::CorrelationFactor& cf, size_t max);

    void compute(size_t J, double T);
};

class F12Squared : public GaussianFundamental
{
public:
    F12Squared(const parameters::CorrelationFactor& cf, size_t max);

    void compute(size_t J, double T);
};

class F12G12 : public GaussianFundamental
{
private:
    FJT Fm_;
public:
    F12G12(const parameters::CorrelationFactor& cf, size_t max);

    void compute(size_t J, double T);
};

class F12DoubleCommutator : public GaussianFundamental
{
public:
    F12DoubleCommutator(const parameters::CorrelationFactor& cf, size_t max);

    void compute(size_t J, double T);
};

class ErfFundamental : public base::Fjt
{
private:
    double omega_;
    FJT boys_;

public:
    ErfFundamental(double omega, size_t max);

    void compute(size_t J, double T);

    void set_omega(double omega)
    {
        omega_ = omega;
    }
};

class ErfComplementFundamental : public base::Fjt
{
private:
    double omega_;
    FJT boys_;

public:
    ErfComplementFundamental(double omega, size_t max);

    void compute(size_t J, double T);

    void set_omega(double omega)
    {
        omega_ = omega;
    }

};

}

} // namespace lightspeed

#endif // header guard
