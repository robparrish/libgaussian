#ifndef CONSTANTS_H
#define CONSTANTS_H

namespace lightspeed {

namespace constants {

extern const int max_df__;
extern const double df__[];

extern const int max_fac__;
extern const double fac__[];

constexpr int ncartesian(int am)
{
    return (am >= 0) ? ((((am) + 2) * ((am) + 1)) >> 1) : 0;
}
constexpr int nspherical(int am)
{
    return (am >= 0) ? (2 * am + 1) : 0;
}

}

}

#endif
