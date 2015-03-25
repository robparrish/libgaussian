#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <cstddef>

namespace lightspeed {

namespace constants {

constexpr int max_df__ = 300;
extern const double df__[];

constexpr int max_fac__ = 50;
extern const double fac__[];

constexpr int max_bc__ = 20;
extern const double bc__[max_bc__][max_bc__];

constexpr size_t ncartesian(size_t am)
{
    return (am > 0) ? ((((am) + 2) * ((am) + 1)) >> 1) : 0;
}
constexpr size_t nspherical(size_t am)
{
    return (am > 0) ? (2 * am + 1) : 0;
}

}

}

#endif
