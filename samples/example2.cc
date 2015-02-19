#include <cstdlib>
#include <cstdio>
#include <core/am.h>
#include <vector>

using namespace libgaussian;

int main(int argc, char* argv[])
{
    int Lmax = 4;

    std::vector<SAngularMomentum> am_info = SAngularMomentum::build(Lmax);;
    for (int L = 0; L <= Lmax; L++) {
        const SAngularMomentum& am = am_info[L];
        for (size_t ind = 0; ind < am.ncoef(); ind++) {
            printf("L   %1d Pure ID %3d Cart ID %3d Val %24.16E\n",
                L,
                am.spherical_ind(ind),
                am.cartesian_ind(ind),
                am.cartesian_coef(ind));
        }
    }

    return EXIT_SUCCESS;
}

