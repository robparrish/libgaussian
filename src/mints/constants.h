#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <cstddef>

namespace lightspeed {

namespace constants {

extern const int max_df__;
extern const double df__[];

extern const int max_fac__;
extern const double fac__[];

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
