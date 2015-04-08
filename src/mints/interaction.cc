#include "interaction.h"
#include "boys.h"
#include <string.h>
#include <math.h>

namespace lightspeed {

/**
 * Standard parameters for Klopper's Slater-type geminal, approximated as a
 * six-term contracted Gaussian-type geminal.
 * 
 * TODO: Figure out how to use this
 **/
namespace klopper {

/// Standard Contraction Coefficients of Klopper's Slater
std::vector<double> slater_ws = {
-0.3144,
-0.3037,
-0.1681,
-0.09811,
-0.06024,
-0.03726
};

/// Standard Exponential Coefficients of Klopper's Slater
std::vector<double> slater_cs = {
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

EwaldInteraction::EwaldInteraction(
    double a, 
    double b,
    double w) :
    Interaction(),
    a_(a),
    b_(b),
    w_(w)
{
    parameters_["a"] = a_;
    parameters_["b"] = b_;
    parameters_["w"] = w_;
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

} // namespace lightspeed
