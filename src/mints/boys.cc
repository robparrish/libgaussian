#include "boys.h"
#include "constants.h"
#include <math.h>

namespace lightspeed {

#define EPS 1.0e-17

void Boys::compute(double t, int n)
{
    int i, m;
    int m2;
    double t2;
    double num;
    double sum;
    double term1;
    static double K = 1.0 / M_2_SQRTPI;
    double et;

    if (t > 20.0) {
        t2 = 2 * t;
        et = exp(-t);
        t = sqrt(t);
        buffer_[0] = K * erf(t) / t;
        for (m = 0; m <= n - 1; m++) {
            buffer_[m + 1] = ((2 * m + 1) * buffer_[m] - et) / (t2);
        }
    }
    else {
        et = exp(-t);
        t2 = 2 * t;
        m2 = 2 * n;
        num = constants::df__[m2];
        i = 0;
        sum = 1.0 / (m2 + 1);
        do {
            i++;
            num = num * t2;
            term1 = num / constants::df__[m2 + 2 * i + 2];
            sum += term1;
        } while (fabs(term1) > EPS && i < constants::max_df__);
        buffer_[n] = sum * et;
        for (m = n - 1; m >= 0; m--) {
            buffer_[m] = (t2 * buffer_[m + 1] + et) / (2 * m + 1);
        }
    }
}

} // namespace lightspeed
