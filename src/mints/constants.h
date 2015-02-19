#ifndef CONSTANTS_H
#define CONSTANTS_H

namespace libgaussian {

namespace constants {

extern const int max_df__;
extern const double df__[];

constexpr int ncartesian(int am)
{
    return (am >= 0) ? ((((am) + 2) * ((am) + 1)) >> 1) : 0;
}

}

}

#endif
