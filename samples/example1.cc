#include <cstdlib>
#include <cstdio>
#include <vector>
#include <math.h>
#include <ambit/tensor.h>
#include <core/am.h>
#include <core/basisset.h>
#include <core/molecule.h>
#include <mints/int2c.h>
#include <mints/int4c.h>

using namespace ambit;
using namespace lightspeed;

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
std::shared_ptr<SBasisSet> get_h2o_sto3g(bool spherical)
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
            SGaussianShell(O1.x(),O1.y(),O1.z(),spherical,0,O1s_c,O1s_e),
            SGaussianShell(O1.x(),O1.y(),O1.z(),spherical,0,O2s_c,O2s_e),
            SGaussianShell(O1.x(),O1.y(),O1.z(),spherical,1,O2p_c,O2p_e)
        },
        {
            SGaussianShell(H2.x(),H2.y(),H2.z(),spherical,0,H1s_c,H1s_e)
        },
        {
            SGaussianShell(H3.x(),H3.y(),H3.z(),spherical,0,H1s_c,H1s_e)
        },
        };

    return std::shared_ptr<SBasisSet>(new SBasisSet(name,shells));
}
int main(int argc, char* argv[])
{
    std::shared_ptr<SMolecule> mol = get_h2o();
    std::shared_ptr<SBasisSet> bas = get_h2o_sto3g(false);
    size_t nbf  = bas->nfunction();
    size_t nocc = 5;

    mol->print();
    bas->print();

    OverlapInt2C Sints(bas,bas);
    double* Sbuffer = Sints.buffer();
    Tensor S = Tensor::build(kCore, "S", {nbf, nbf});
    double* Sp = S.data().data();
    for (int P = 0; P < bas->nshell(); P++) {
        for (int Q = 0; Q < bas->nshell(); Q++) {
            Sints.compute_shell(P,Q);
            int oP = bas->shell(P).function_index();
            int oQ = bas->shell(Q).function_index();
            int nP = bas->shell(P).nfunction();
            int nQ = bas->shell(Q).nfunction();
            for (int p = 0, index = 0; p < nP; p++) {
                for (int q = 0; q < nQ; q++, index++) {
                    Sp[(p + oP) * nbf + (q + oQ)] = Sbuffer[index];
                }
            }
        }
    }
    //S.print();

    KineticInt2C Tints(bas,bas);
    double* Tbuffer = Tints.buffer();
    Tensor T = Tensor::build(kCore, "T", {nbf, nbf});
    double* Tp = T.data().data();
    for (int P = 0; P < bas->nshell(); P++) {
        for (int Q = 0; Q < bas->nshell(); Q++) {
            Tints.compute_shell(P,Q);
            int oP = bas->shell(P).function_index();
            int oQ = bas->shell(Q).function_index();
            int nP = bas->shell(P).nfunction();
            int nQ = bas->shell(Q).nfunction();
            for (int p = 0, index = 0; p < nP; p++) {
                for (int q = 0; q < nQ; q++, index++) {
                    Tp[(p + oP) * nbf + (q + oQ)] = Tbuffer[index];
                }
            }
        }
    }
    //T.print();

    DipoleInt2C Xints(bas,bas);
    size_t Xchunk = Xints.chunk_size();
    double* Xbuffer = Xints.buffer();
    double* Ybuffer = Xbuffer + Xints.chunk_size();
    double* Zbuffer = Ybuffer + Xints.chunk_size();
    std::vector<Tensor> Xs(3);
    Xs[0] = Tensor::build(kCore, "X", {nbf, nbf});
    Xs[1] = Tensor::build(kCore, "Y", {nbf, nbf});
    Xs[2] = Tensor::build(kCore, "Z", {nbf, nbf});
    double* Xp = Xs[0].data().data();
    double* Yp = Xs[1].data().data();
    double* Zp = Xs[2].data().data();
    for (int P = 0; P < bas->nshell(); P++) {
        for (int Q = 0; Q < bas->nshell(); Q++) {
            Xints.compute_shell(P,Q);
            int oP = bas->shell(P).function_index();
            int oQ = bas->shell(Q).function_index();
            int nP = bas->shell(P).nfunction();
            int nQ = bas->shell(Q).nfunction();
            for (int p = 0, index = 0; p < nP; p++) {
                for (int q = 0; q < nQ; q++, index++) {
                    Xp[(p + oP) * nbf + (q + oQ)] = Xbuffer[index];
                    Yp[(p + oP) * nbf + (q + oQ)] = Ybuffer[index];
                    Zp[(p + oP) * nbf + (q + oQ)] = Zbuffer[index];
                }
            }
        }
    }
    //Xs[0].print();
    //Xs[1].print();
    //Xs[2].print();

    PotentialInt2C Vints(bas,bas);
    Vints.set_nuclear_potential(mol);
    double* Vbuffer = Vints.buffer();
    Tensor V = Tensor::build(kCore, "V", {nbf, nbf});
    double* Vp = V.data().data();
    for (int P = 0; P < bas->nshell(); P++) {
        for (int Q = 0; Q < bas->nshell(); Q++) {
            Vints.compute_shell(P,Q);
            int oP = bas->shell(P).function_index();
            int oQ = bas->shell(Q).function_index();
            int nP = bas->shell(P).nfunction();
            int nQ = bas->shell(Q).nfunction();
            for (int p = 0, index = 0; p < nP; p++) {
                for (int q = 0; q < nQ; q++, index++) {
                    Vp[(p + oP) * nbf + (q + oQ)] = Vbuffer[index];
                }
            }
        }
    }
    //V.print();

    PotentialInt4C Iints(bas,bas,bas,bas);
    double* Ibuffer = Iints.buffer();
    Tensor I = Tensor::build(kCore, "I", {nbf,nbf,nbf,nbf});
    double* Ip = I.data().data();
    for (int P = 0; P < bas->nshell(); P++) {
    for (int Q = 0; Q < bas->nshell(); Q++) {
    for (int R = 0; R < bas->nshell(); R++) {
    for (int S = 0; S < bas->nshell(); S++) {
        Iints.compute_shell(P,Q,R,S);
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
            Ip[(p + oP) * nbf * nbf * nbf + (q + oQ) * nbf * nbf + (r + oR) * nbf + (s + oS)] = Ibuffer[index++];
        }}}}
    }}}}
    //I.print();

    // Hartree-Fock
    Tensor X = S.power(-0.5);
    Tensor H = Tensor::build(kCore, "H", {nbf,nbf});  
    Tensor Ft = Tensor::build(kCore, "Ft", {nbf,nbf});  
    Tensor C = Tensor::build(kCore, "C", {nbf,nbf});  
    Tensor Cocc = Tensor::build(kCore, "Cocc", {nbf,nocc});
    Tensor D = Tensor::build(kCore, "D", {nbf,nbf});  
    Tensor F = Tensor::build(kCore, "F", {nbf,nbf}); 

    H() = T();
    H() += V();

    F() = H();
    Ft("ij") = X("pi") * H("pq") * X("qj");
    auto Feig = Ft.syev(kAscending); 
    C("pi") = X("pj") * Feig["eigenvectors"]("ij");
    Cocc() = C({{0,nbf},{0,nocc}});
    D("pq") = Cocc("pi") * Cocc("qi");

    bool converged = false;
    double Enuc = mol->nuclear_repulsion_energy();
    double Eelec = D("pq") * (H("pq") + F("pq"));
    double Eold = 0.0;
    int iter = 1;
    printf("  @RHF iter %5d: %20.14lf\n", iter++, Enuc + Eelec);
    do {

        F("pq") = H("pq");
        F("pq") += D("rs") * (2.0 * I("pqrs") - I("prsq"));
        Eelec = D("pq") * (H("pq") + F("pq"));

        printf("  @RHF iter %5d: %20.14lf\n", iter++, Enuc + Eelec);

        Ft("ij") = X("pi") * F("pq") * X("qj");
        auto Feig = Ft.syev(kAscending); 
        C("pi") = X("pj") * Feig["eigenvectors"]("ij");
        Cocc() = C({{0,nbf},{0,nocc}});
        D("pq") = Cocc("pi") * Cocc("qi");

        if (fabs(Eelec - Eold) < 1.0e-8) converged = true;
        Eold = Eelec;

        if (iter > 15)
            break;
    } while (!converged);

    return EXIT_SUCCESS;
}

