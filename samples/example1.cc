#include <cstdlib>
#include <cstdio>
#include <vector>
#include <core/am.h>
#include <core/basisset.h>
#include <core/molecule.h>
#include <mints/int2c.h>
#include <mints/int4c.h>

using namespace libgaussian;

std::shared_ptr<SMolecule> get_h2o()
{
    std::vector<double> O1r = {0.0000000000000000E+00,  0.0000000000000000E+00, -1.2947689015702168E-01};
    std::vector<double> H2r = {0.0000000000000000E+00, -1.4941867505039508E+00,  1.0274461029282449E+00};
    std::vector<double> H3r = {0.0000000000000000E+00,  1.4941867505039510E+00,  1.0274461029282449E+00};

    std::string name = "H2O";
    std::vector<SAtom> atoms = {
        SAtom("O1","O",8,O1r[0],O1r[1],O1r[2],8.0,4.0,4.0),
        SAtom("H2","H",1,H2r[0],H2r[1],H2r[2],1.0,0.5,0.5),
        SAtom("H3","H",1,H3r[0],H3r[1],H3r[2],1.0,0.5,0.5)};
    return std::shared_ptr<SMolecule>(new SMolecule(name,atoms));
}
std::shared_ptr<SBasisSet> get_h2o_sto3g()
{
    std::vector<double> O1s_c = { 4.2519432829437198E+00,  4.1122937184311832E+00,  1.2816225325813408E+00};
    std::vector<double> O1s_e = { 1.3070931999999999E+02,  2.3808861000000000E+01,  6.4436083000000002E+00};

    std::vector<double> O2s_c = {-2.3941300299447710E-01,  3.2023422913389127E-01,  2.4168557075321612E-01};
    std::vector<double> O2s_e = { 5.0331513000000001E+00,  1.1695960999999999E+00,  3.8038899999999998E-01};

    std::vector<double> O2p_c = { 1.6754501181141905E+00,  1.0535680079948457E+00,  1.6690289807574885E-01};
    std::vector<double> O2p_e = { 5.0331513000000001E+00,  1.1695960999999999E+00,  3.8038899999999998E-01};

    std::vector<double> H1s_c = { 2.7693436095264762E-01,  2.6783885175252709E-01,  8.3473669616925303E-02};
    std::vector<double> H1s_e = { 3.4252509099999999E+00,  6.2391373000000006E-01,  1.6885539999999999E-01};

    std::shared_ptr<SMolecule> mol = get_h2o();

    const SAtom& O1 = mol->atom(0);
    const SAtom& H2 = mol->atom(1);
    const SAtom& H3 = mol->atom(2);

    std::string name = "STO-3G";
    std::vector<std::vector<SGaussianShell>> shells = {
        {
            SGaussianShell(O1.x(),O1.y(),O1.z(),false,0,O1s_c,O1s_e),
            SGaussianShell(O1.x(),O1.y(),O1.z(),false,0,O2s_c,O2s_e),
            SGaussianShell(O1.x(),O1.y(),O1.z(),false,1,O2p_c,O2p_e)
        },
        {
            SGaussianShell(H2.x(),H2.y(),H2.z(),false,0,H1s_c,H1s_e)
        },
        {
            SGaussianShell(H3.x(),H3.y(),H3.z(),false,0,H1s_c,H1s_e)
        },
        };

    return std::shared_ptr<SBasisSet>(new SBasisSet(name,shells));
}
int main(int argc, char* argv[])
{
    std::shared_ptr<SMolecule> mol = get_h2o();
    std::shared_ptr<SBasisSet> bas = get_h2o_sto3g();

    mol->print(stdout,true);

    printf("Overlap Ints:\n\n");
    OverlapInt2C Sints(bas,bas);
    double* Sbuffer = Sints.buffer();
    for (int P = 0; P < bas->nshell(); P++) {
        for (int Q = 0; Q < bas->nshell(); Q++) {
            Sints.compute_pair(P,Q);
            int oP = bas->shell(P).function_index();
            int oQ = bas->shell(Q).function_index();
            int nP = bas->shell(P).nfunction();
            int nQ = bas->shell(Q).nfunction();
            for (int p = 0, index = 0; p < nP; p++) {
                for (int q = 0; q < nQ; q++, index++) {
                    printf("%3d %3d %24.16E\n", p + oP, q + oQ, Sbuffer[index]);
                }
            }
        }
    }
    printf("\n");

    printf("Kinetic Ints:\n\n");
    KineticInt2C Tints(bas,bas);
    double* Tbuffer = Tints.buffer();
    for (int P = 0; P < bas->nshell(); P++) {
        for (int Q = 0; Q < bas->nshell(); Q++) {
            Tints.compute_pair(P,Q);
            int oP = bas->shell(P).function_index();
            int oQ = bas->shell(Q).function_index();
            int nP = bas->shell(P).nfunction();
            int nQ = bas->shell(Q).nfunction();
            for (int p = 0, index = 0; p < nP; p++) {
                for (int q = 0; q < nQ; q++, index++) {
                    printf("%3d %3d %24.16E\n", p + oP, q + oQ, Tbuffer[index]);
                }
            }
        }
    }
    printf("\n");

    printf("Dipole Ints:\n\n");
    DipoleInt2C Xints(bas,bas);
    size_t Xchunk = Xints.chunk_size();
    double* Xbuffer = Xints.buffer();
    printf("X:\n\n");
    for (int P = 0; P < bas->nshell(); P++) {
        for (int Q = 0; Q < bas->nshell(); Q++) {
            Xints.compute_pair(P,Q);
            int oP = bas->shell(P).function_index();
            int oQ = bas->shell(Q).function_index();
            int nP = bas->shell(P).nfunction();
            int nQ = bas->shell(Q).nfunction();
            for (int p = 0, index = 0; p < nP; p++) {
                for (int q = 0; q < nQ; q++, index++) {
                    printf("%3d %3d %24.16E\n", p + oP, q + oQ, Xbuffer[index + 0L * Xchunk]);
                }
            }
        }
    }

    printf("Y:\n\n");
    for (int P = 0; P < bas->nshell(); P++) {
        for (int Q = 0; Q < bas->nshell(); Q++) {
            Xints.compute_pair(P,Q);
            int oP = bas->shell(P).function_index();
            int oQ = bas->shell(Q).function_index();
            int nP = bas->shell(P).nfunction();
            int nQ = bas->shell(Q).nfunction();
            for (int p = 0, index = 0; p < nP; p++) {
                for (int q = 0; q < nQ; q++, index++) {
                    printf("%3d %3d %24.16E\n", p + oP, q + oQ, Xbuffer[index + 1L * Xchunk]);
                }
            }
        }
    }

    printf("Z:\n\n");
    for (int P = 0; P < bas->nshell(); P++) {
        for (int Q = 0; Q < bas->nshell(); Q++) {
            Xints.compute_pair(P,Q);
            int oP = bas->shell(P).function_index();
            int oQ = bas->shell(Q).function_index();
            int nP = bas->shell(P).nfunction();
            int nQ = bas->shell(Q).nfunction();
            for (int p = 0, index = 0; p < nP; p++) {
                for (int q = 0; q < nQ; q++, index++) {
                    printf("%3d %3d %24.16E\n", p + oP, q + oQ, Xbuffer[index + 2L * Xchunk]);
                }
            }
        }
    }
    printf("\n");

    printf("Electron Repulsion Ints:\n\n");
    PotentialInt4C Iints(bas,bas,bas,bas);
    double* Ibuffer = Iints.buffer();
    for (int P = 0; P < bas->nshell(); P++) {
    for (int Q = 0; Q < bas->nshell(); Q++) {
    for (int R = 0; R < bas->nshell(); R++) {
    for (int S = 0; S < bas->nshell(); S++) {
        Iints.compute_quartet(P,Q,R,S);
        int oP = bas->shell(P).function_index();
        int oQ = bas->shell(Q).function_index();
        int oR = bas->shell(R).function_index();
        int oS = bas->shell(S).function_index();
        int nP = bas->shell(P).nfunction();
        int nQ = bas->shell(Q).nfunction();
        int nR = bas->shell(R).nfunction();
        int nS = bas->shell(S).nfunction();
        int index = 0;
        for (int p = 0; p < nP; p++) {
        for (int q = 0; q < nQ; q++) {
        for (int r = 0; r < nR; r++) {
        for (int s = 0; s < nS; s++) {
            printf("%3d %3d %3d %3d %24.16E\n", p + oP, q + oQ, r + oR, s + oS, Ibuffer[index++]);
        }}}}
    }}}}
    printf("\n");


    return EXIT_SUCCESS;
}

