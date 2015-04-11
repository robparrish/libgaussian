#include "interaction.h"
#include "boys.h"
#include <string.h>
#include <math.h>
#include <cassert>

namespace lightspeed {

/**
 * Standard parameters for Klopper's Slater-type geminal, approximated as a
 * six-term contracted Gaussian-type geminal.
 *
 * TODO: Figure out how to use this
 **/
namespace klopper {

/// Standard Contraction Coefficients of Klopper's Slater
std::vector<double> slater_cs = {
-0.3144,
-0.3037,
-0.1681,
-0.09811,
-0.06024,
-0.03726
};

/// Standard Exponential Coefficients of Klopper's Slater
std::vector<double> slater_es = {
0.2209,
1.004,
3.622,
12.16,
45.87,
254.4
};

}

std::shared_ptr<Interaction> Interaction::coulomb()
{
    return std::make_shared<EwaldInteraction>(1.0,0.0,0.0);
}
std::shared_ptr<Interaction> Interaction::ewald(double a, double b, double w)
{
    return std::make_shared<EwaldInteraction>(a,b,w);
}
std::shared_ptr<Interaction> Interaction::f12(double slater_exponent)
{
    return std::make_shared<F12Interaction>(slater_exponent);
}

EwaldInteraction::EwaldInteraction(
    double a,
    double b,
    double w) :
    Interaction(),
    a_(a),
    b_(b),
    w_(w)
{
    parameters_["a"] = std::to_string(a_);
    parameters_["b"] = std::to_string(b_);
    parameters_["w"] = std::to_string(w_);
    name_ = "Ewald";
    description_ = "a/r + b erf(wr)/r";
}
void EwaldInteraction::initialize_fundamental(int Lmax)
{
    Interaction::initialize_fundamental(Lmax);
    boys_ = std::make_shared<Boys>(Lmax);
    boys_buffer_ = boys_->buffer().data();
}
void EwaldInteraction::compute_fundamental(double rho, double T, int L)
{
    for (int i = 0; i <= L; i++) {
        buffer_[i] = 0.0;
    }

    double pref = 2.0 * M_PI / rho;

    if (a_ != 0.0) {
        boys_->compute(T,L);
        for (int i = 0; i <= L; i++) {
            buffer_[i] += a_ * pref * boys_buffer_[i];
        }
    }

    if (b_ != 0.0) {
        double s = w_ * w_ / (w_ * w_ + rho);
        boys_->compute(s * T, L);
        for (int i = 0; i <= L; i++) {
            buffer_[i] += b_ * pref * sqrt(s) * boys_buffer_[i];
        }
    }
}

F12Interaction::F12Interaction(double slater_exponent)
    : slater_exponent_(slater_exponent), cs_(klopper::slater_cs), es_(klopper::slater_es)
{
    assert(cs_.size() == es_.size());
    nparam_ = cs_.size();

    for (double& e : es_)
        e *= slater_exponent_ * slater_exponent_;

    parameters_["Slater exponent"] = std::to_string(slater_exponent_);
    parameters_["Gaussian geminal type"] = "Klopper 6-parameter";
    name_ = "F12";
    description_ = "";
}

F12Interaction::F12Interaction(const std::vector<double>& cs, const std::vector<double>& es)
    : slater_exponent_(1.0), cs_(cs), es_(es)
{
    assert(cs_.size() == es_.size());
    nparam_ = cs_.size();

    parameters_["Slater exponent"] = std::to_string(slater_exponent_);
    parameters_["Gaussian geminal type"] = "User specified";
    name_ = "F12";
    description_ = "";
}

void F12Interaction::initialize_fundamental(int Lmax)
{
    Interaction::initialize_fundamental(Lmax);
}

void F12Interaction::compute_fundamental(double rho, double T, int L)
{
    for (int l=0; l<=L; ++l)
        buffer_[l] = 0.0;

    double pfac, expterm, rhotilde, omega;
    double eri_correct = rho / 2 / M_PI;
    for (size_t i=0; i<nparam_; ++i) {
        omega = es_[i];
        rhotilde = omega / (rho + omega);
        pfac = cs_[i] * pow(M_PI/(rho + omega), 1.5) * eri_correct;
        expterm = exp(-rhotilde*T)*pfac;
        for (int l=0; l<=L; ++l) {
            buffer_[l] += expterm;
            expterm *= rhotilde;
        }
    }
}

} // namespace lightspeed
