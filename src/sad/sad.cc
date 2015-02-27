#include "sad.h"
#include <math.h>
#include <core/molecule.h>
#include <core/basisset.h>
#include <mints/int2c.h>

#include <omp.h>

using namespace ambit;

namespace lightspeed {

SAD::SAD(
    const std::shared_ptr<SMolecule>& molecule,
    const std::shared_ptr<SBasisSet>& primary,
    const std::shared_ptr<SBasisSet>& minao) :
    molecule_(molecule),
    primary_(primary),
    minao_(minao)
{
    // => Validate Setup <= //

    // Check number of atoms
    if (molecule_->natom() != primary_->natom()) throw std::runtime_error("SAD: natom does not match.");
    if (molecule_->natom() != minao_->natom()) throw std::runtime_error("SAD: natom does not match.");

    for (size_t A = 0; A < molecule_->natom(); A++) {

        const SAtom& atom = molecule_->atom(A);
        int N = atom.N();

        // Dummy atoms are ignored (e.g., midbond functions)
        if (N == 0) continue;
        // Ghost atoms are ignored
        if (atom.Ya() == 0.0 && atom.Yb() == 0.0) continue;
        
        // Check primary basis set centers
        for (size_t P2 = 0; P2 < primary_->atoms_to_shell_inds()[A].size(); P2++) {
            const SGaussianShell& shell = primary_->shell(primary_->atoms_to_shell_inds()[A][P2]);
            if (atom.x() != shell.x()) throw std::runtime_error("SAD: shell position does not match for atom: " + atom.label());
            if (atom.y() != shell.y()) throw std::runtime_error("SAD: shell position does not match for atom: " + atom.label());
            if (atom.z() != shell.z()) throw std::runtime_error("SAD: shell position does not match for atom: " + atom.label());
        }

        // Check minao basis set centers
        for (size_t P2 = 0; P2 < minao_->atoms_to_shell_inds()[A].size(); P2++) {
            const SGaussianShell& shell = minao_->shell(minao_->atoms_to_shell_inds()[A][P2]);
            if (atom.x() != shell.x()) throw std::runtime_error("SAD: shell position does not match for atom: " + atom.label());
            if (atom.y() != shell.y()) throw std::runtime_error("SAD: shell position does not match for atom: " + atom.label());
            if (atom.z() != shell.z()) throw std::runtime_error("SAD: shell position does not match for atom: " + atom.label());
        }

        //int nocc;
        //int nfrz;
        //int nact;
        //std::vector<int> nshell_by_am;
        std::vector<int> am_types; 
        if (N <=2) {
            //nocc = 1; // 1s
            //nfrz = 0; // 
            //nact = 1; // 1s
            //nshell_by_am = {1};
            am_types = {0};    
        } else if (N <= 10) {
            //nocc = 5; // 1s 2s 2p
            //nfrz = 1; // 1s
            //nact = 4; // 2s 2p
            //nshell_by_am = {2,1};
            am_types = {0,0,1};    
        } else if (N <= 18) {
            //nocc = 9; // 1s 2s 2p 3s 3p
            //nfrz = 5; // 1s 2s 2p
            //nact = 4; // 3s 3p
            //nshell_by_am = {3,2};
            am_types = {0,0,0,1,1};    
        } else if (N <= 36) {
            //nocc = 18; // 1s 2s 2p 3s 3p 4s 3d 4p
            //nfrz = 9;  // 1s 2s 2p 3s 3p
            //nact = 9;  // 4s 3d 4p
            //nshell_by_am = {4,3,1};
            am_types = {0,0,0,0,1,1,1,2};    
        } else if (N <= 54) {
            //nocc = 27; // 1s 2s 2p 3s 3p 4s 3d 4p 5s 4d 5p
            //nfrz = 18; // 1s 2s 2p 3s 3p 4s 3d 4p
            //nact = 9;  // 5s 4d 5p
            //nshell_by_am = {5,4,2};
            am_types = {0,0,0,0,0,1,1,1,1,2,2};    
        } else if (N <= 86) {
            //nocc = 43; // 1s 2s 2p 3s 3p 4s 3d 4p 5s 4d 5p 6s 4f 5d 6p
            //nfrz = 27; // 1s 2s 2p 3s 3p 4s 3d 4p 5s 4d 5p
            //nact = 16; // 6s 4f 5d 6p
            //nshell_by_am = {6,5,3,1};
            am_types = {0,0,0,0,0,0,1,1,1,1,1,2,2,2,3};    
        } else if (N <= 118) {
            //nocc = 59; // 1s 2s 2p 3s 3p 4s 3d 4p 5s 4d 5p 6s 4f 5d 6p 7s 5f 6d 7p
            //nfrz = 43; // 1s 2s 2p 3s 3p 4s 3d 4p 5s 4d 5p 6s 4f 5d 6p
            //nact = 16; // 7s 5f 6d 7p
            //nshell_by_am = {7,6,4,2};
            am_types = {0,0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,3,3};    
        } else {
            throw std::runtime_error("SAD: N > 118 not supported.");
        }

        if (minao_->atoms_to_shell_inds()[A].size() != am_types.size()) throw std::runtime_error("SAD: MinAO basis has wrong number of shells for atom: " + atom.label());
        
        for (size_t P2 = 0; P2 < minao_->atoms_to_shell_inds()[A].size(); P2++) {
            const SGaussianShell& shell = minao_->shell(minao_->atoms_to_shell_inds()[A][P2]);
            if (am_types[P2] != shell.am()) throw std::runtime_error("SAD: AM types do not match for atom: " + atom.label());
            if (!shell.is_spherical()) throw std::runtime_error("SAD: MinAO Basis is not spherical for atom: " + atom.label());
        } 
    }
}
void SAD::print(FILE* fh) const
{
    fprintf(fh, "  SAD:\n");
    fprintf(fh, "    Molecule      = %18s\n", molecule_->name().c_str());
    fprintf(fh, "    Primary Basis = %18s\n", primary_->name().c_str());
    fprintf(fh, "    MinAO Basis   = %18s\n", minao_->name().c_str());
    fprintf(fh, "\n");
}
Tensor SAD::compute_C() const
{
    return compute_C_helper("C");
}
Tensor SAD::compute_Ca() const
{
    return compute_C_helper("CA");
}
Tensor SAD::compute_Cb() const
{
    return compute_C_helper("CB");
}
Tensor SAD::compute_C_helper(const std::string& key) const
{
    size_t nbf = primary_->nfunction();
    size_t nocc = 0;
    std::vector<Tensor> atom_blocks;
    std::vector<size_t> offsets;
    for (size_t A = 0; A < molecule_->natom(); A++) {

        const SAtom& atom = molecule_->atom(A);
        int N = atom.N();

        // Dummy atoms are ignored (e.g., midbond functions)
        if (N == 0) continue;
        // Ghost atoms are ignored
        if (atom.Ya() == 0.0 && atom.Yb() == 0.0) continue;
        
        double Q = 0.0;
        if (key == "C") {
            Q = 0.5 * (atom.Ya() + atom.Yb()); 
        } else if (key == "CA") {
            Q = atom.Ya();
        } else if (key == "CB") {
            Q = atom.Yb();
        } else {
            throw std::runtime_error("SAD: unknown key.");
        }

        Tensor C = compute_atom(A,Q);
        nocc += C.dim(1);
        atom_blocks.push_back(C);
        offsets.push_back(primary_->shell(primary_->atoms_to_shell_inds()[A][0]).function_index());
    }
    
    Tensor L = Tensor::build(kCore,"C (SAD)", {nbf,nocc});
    for (size_t ind = 0, occstart = 0; ind < atom_blocks.size(); ind++) {
        Tensor C = atom_blocks[ind];
        size_t nbfstart = offsets[ind];
        size_t nbfA = C.dim(0);
        size_t noccA = C.dim(1);
        L({{nbfstart,nbfstart+nbfA},{occstart,occstart+noccA}}) = C();
        occstart += noccA; // FMS
    }
    return L;
}
Tensor SAD::compute_atom(
    size_t A,
    double Q) const
{
    const SAtom& atom = molecule_->atom(A);
    int N = atom.N();

    if (N == 0) throw std::runtime_error("SAD: Do not call this on dummy atoms");

    // => Occupation <= //

    int nocc;
    int nfrz;
    int nact;
    //std::vector<int> nshell_by_am;
    //std::vector<int> am_types; 
    if (N <=2) {
        nocc = 1; // 1s
        nfrz = 0; // 
        nact = 1; // 1s
        //nshell_by_am = {1};
        //am_types = {0};    
    } else if (N <= 10) {
        nocc = 5; // 1s 2s 2p
        nfrz = 1; // 1s
        nact = 4; // 2s 2p
        //nshell_by_am = {2,1};
        //am_types = {0,0,1};    
    } else if (N <= 18) {
        nocc = 9; // 1s 2s 2p 3s 3p
        nfrz = 5; // 1s 2s 2p
        nact = 4; // 3s 3p
        //nshell_by_am = {3,2};
        //am_types = {0,0,0,1,1};    
    } else if (N <= 36) {
        nocc = 18; // 1s 2s 2p 3s 3p 4s 3d 4p
        nfrz = 9;  // 1s 2s 2p 3s 3p
        nact = 9;  // 4s 3d 4p
        //nshell_by_am = {4,3,1};
        //am_types = {0,0,0,0,1,1,1,2};    
    } else if (N <= 54) {
        nocc = 27; // 1s 2s 2p 3s 3p 4s 3d 4p 5s 4d 5p
        nfrz = 18; // 1s 2s 2p 3s 3p 4s 3d 4p
        nact = 9;  // 5s 4d 5p
        //nshell_by_am = {5,4,2};
        //am_types = {0,0,0,0,0,1,1,1,1,2,2};    
    } else if (N <= 86) {
        nocc = 43; // 1s 2s 2p 3s 3p 4s 3d 4p 5s 4d 5p 6s 4f 5d 6p
        nfrz = 27; // 1s 2s 2p 3s 3p 4s 3d 4p 5s 4d 5p
        nact = 16; // 6s 4f 5d 6p
        //nshell_by_am = {6,5,3,1};
        //am_types = {0,0,0,0,0,0,1,1,1,1,1,2,2,2,3};    
    } else if (N <= 118) {
        nocc = 59; // 1s 2s 2p 3s 3p 4s 3d 4p 5s 4d 5p 6s 4f 5d 6p 7s 5f 6d 7p
        nfrz = 43; // 1s 2s 2p 3s 3p 4s 3d 4p 5s 4d 5p 6s 4f 5d 6p
        nact = 16; // 7s 5f 6d 7p
        //nshell_by_am = {7,6,4,2};
        //am_types = {0,0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,3,3};    
    } else {
        throw std::runtime_error("SAD: N > 118 not supported.");
    }

    // Fill the frozen orbitals fully, then fractionally occupy the active orbitals
    double focc = 1.0;
    double aocc = (Q - nfrz) / nact;
    // Unless the user is an idiot in which case fractionally occupy the core
    if (Q < nfrz) {
        focc = Q / nfrz;
        aocc = 0.0;
    }    

    //printf("Atom %4s: Q = %11.3E, focc = %11.3E aocc = %11.3E\n", atom.label().c_str(), Q, focc, aocc);

    Tensor f = Tensor::build(kCore,"f",{nocc});
    int i = 0;
    double* fp = f.data().data();
    if (N <=2) {
        fp[i++] = aocc; // 1s
    } else if (N <= 10) {
        fp[i++] = focc; // 1s
        fp[i++] = aocc; // 2s
        fp[i++] = aocc; // 2p
        fp[i++] = aocc; // 2p
        fp[i++] = aocc; // 2p
    } else if (N <= 18) {
        fp[i++] = focc; // 1s
        fp[i++] = focc; // 2s
        fp[i++] = aocc; // 3s
        fp[i++] = focc; // 2p
        fp[i++] = focc; // 2p
        fp[i++] = focc; // 2p
        fp[i++] = aocc; // 3p
        fp[i++] = aocc; // 3p
        fp[i++] = aocc; // 3p
    } else if (N <= 36) {
        fp[i++] = focc; // 1s
        fp[i++] = focc; // 2s
        fp[i++] = focc; // 3s
        fp[i++] = aocc; // 4s
        fp[i++] = focc; // 2p
        fp[i++] = focc; // 2p
        fp[i++] = focc; // 2p
        fp[i++] = focc; // 3p
        fp[i++] = focc; // 3p
        fp[i++] = focc; // 3p
        fp[i++] = aocc; // 4p
        fp[i++] = aocc; // 4p
        fp[i++] = aocc; // 4p
        fp[i++] = aocc; // 3d
        fp[i++] = aocc; // 3d
        fp[i++] = aocc; // 3d
        fp[i++] = aocc; // 3d
        fp[i++] = aocc; // 3d
    } else if (N <= 54) {
        fp[i++] = focc; // 1s
        fp[i++] = focc; // 2s
        fp[i++] = focc; // 3s
        fp[i++] = focc; // 4s
        fp[i++] = aocc; // 5s
        fp[i++] = focc; // 2p
        fp[i++] = focc; // 2p
        fp[i++] = focc; // 2p
        fp[i++] = focc; // 3p
        fp[i++] = focc; // 3p
        fp[i++] = focc; // 3p
        fp[i++] = focc; // 4p
        fp[i++] = focc; // 4p
        fp[i++] = focc; // 4p
        fp[i++] = aocc; // 5p
        fp[i++] = aocc; // 5p
        fp[i++] = aocc; // 5p
        fp[i++] = focc; // 3d
        fp[i++] = focc; // 3d
        fp[i++] = focc; // 3d
        fp[i++] = focc; // 3d
        fp[i++] = focc; // 3d
        fp[i++] = aocc; // 4d
        fp[i++] = aocc; // 4d
        fp[i++] = aocc; // 4d
        fp[i++] = aocc; // 4d
        fp[i++] = aocc; // 4d
    } else if (N <= 86) {
        fp[i++] = focc; // 1s
        fp[i++] = focc; // 2s
        fp[i++] = focc; // 3s
        fp[i++] = focc; // 4s
        fp[i++] = focc; // 5s
        fp[i++] = aocc; // 6s
        fp[i++] = focc; // 2p
        fp[i++] = focc; // 2p
        fp[i++] = focc; // 2p
        fp[i++] = focc; // 3p
        fp[i++] = focc; // 3p
        fp[i++] = focc; // 3p
        fp[i++] = focc; // 4p
        fp[i++] = focc; // 4p
        fp[i++] = focc; // 4p
        fp[i++] = focc; // 5p
        fp[i++] = focc; // 5p
        fp[i++] = focc; // 5p
        fp[i++] = aocc; // 6p
        fp[i++] = aocc; // 6p
        fp[i++] = aocc; // 6p
        fp[i++] = focc; // 3d
        fp[i++] = focc; // 3d
        fp[i++] = focc; // 3d
        fp[i++] = focc; // 3d
        fp[i++] = focc; // 3d
        fp[i++] = focc; // 4d
        fp[i++] = focc; // 4d
        fp[i++] = focc; // 4d
        fp[i++] = focc; // 4d
        fp[i++] = focc; // 4d
        fp[i++] = aocc; // 5d
        fp[i++] = aocc; // 5d
        fp[i++] = aocc; // 5d
        fp[i++] = aocc; // 5d
        fp[i++] = aocc; // 5d
        fp[i++] = aocc; // 4f
        fp[i++] = aocc; // 4f
        fp[i++] = aocc; // 4f
        fp[i++] = aocc; // 4f
        fp[i++] = aocc; // 4f
        fp[i++] = aocc; // 4f
        fp[i++] = aocc; // 4f
    } else if (N <= 118) {
        fp[i++] = focc; // 1s
        fp[i++] = focc; // 2s
        fp[i++] = focc; // 3s
        fp[i++] = focc; // 4s
        fp[i++] = focc; // 5s
        fp[i++] = focc; // 6s
        fp[i++] = aocc; // 7s
        fp[i++] = focc; // 2p
        fp[i++] = focc; // 2p
        fp[i++] = focc; // 2p
        fp[i++] = focc; // 3p
        fp[i++] = focc; // 3p
        fp[i++] = focc; // 3p
        fp[i++] = focc; // 4p
        fp[i++] = focc; // 4p
        fp[i++] = focc; // 4p
        fp[i++] = focc; // 5p
        fp[i++] = focc; // 5p
        fp[i++] = focc; // 5p
        fp[i++] = focc; // 6p
        fp[i++] = focc; // 6p
        fp[i++] = focc; // 6p
        fp[i++] = aocc; // 7p
        fp[i++] = aocc; // 7p
        fp[i++] = aocc; // 7p
        fp[i++] = focc; // 3d
        fp[i++] = focc; // 3d
        fp[i++] = focc; // 3d
        fp[i++] = focc; // 3d
        fp[i++] = focc; // 3d
        fp[i++] = focc; // 4d
        fp[i++] = focc; // 4d
        fp[i++] = focc; // 4d
        fp[i++] = focc; // 4d
        fp[i++] = focc; // 4d
        fp[i++] = focc; // 5d
        fp[i++] = focc; // 5d
        fp[i++] = focc; // 5d
        fp[i++] = focc; // 5d
        fp[i++] = focc; // 5d
        fp[i++] = aocc; // 6d
        fp[i++] = aocc; // 6d
        fp[i++] = aocc; // 6d
        fp[i++] = aocc; // 6d
        fp[i++] = aocc; // 6d
        fp[i++] = focc; // 4f
        fp[i++] = focc; // 4f
        fp[i++] = focc; // 4f
        fp[i++] = focc; // 4f
        fp[i++] = focc; // 4f
        fp[i++] = focc; // 4f
        fp[i++] = focc; // 4f
        fp[i++] = aocc; // 5f
        fp[i++] = aocc; // 5f
        fp[i++] = aocc; // 5f
        fp[i++] = aocc; // 5f
        fp[i++] = aocc; // 5f
        fp[i++] = aocc; // 5f
        fp[i++] = aocc; // 5f
    } else {
        throw std::runtime_error("SAD: N > 118 not supported.");
    }

    // => Apply sqrt of occupation <= //

    for (size_t ind = 0; ind < nocc; ind++) {
        if (fp[ind] < 0.0) throw std::runtime_error("SAD: Negative occupation is not permitted.");
        fp[ind] = sqrt(fp[ind]);
    }

    // => Overlap Integrals <= // 

    const std::vector<size_t>& primary_shells = primary_->atoms_to_shell_inds()[A];
    const std::vector<size_t>& minao_shells = minao_->atoms_to_shell_inds()[A];

    size_t nbf  = 0;
    for (size_t P2 = 0; P2 < primary_shells.size(); P2++) {
        nbf += primary_->shell(primary_shells[P2]).nfunction();
    }

    size_t nmin = 0;
    for (size_t P2 = 0; P2 < minao_shells.size(); P2++) {
        nmin += minao_->shell(minao_shells[P2]).nfunction();
    }

    if (nmin != nocc) throw std::runtime_error("This should be impossible");

    Tensor S11 = Tensor::build(kCore,"Spp",{nbf,nbf}); 
    Tensor S12 = Tensor::build(kCore,"Spm",{nbf,nmin}); 
    Tensor S22 = Tensor::build(kCore,"Smm",{nmin,nmin}); 

    double* S11p = S11.data().data();
    double* S12p = S12.data().data();
    double* S22p = S22.data().data();

    OverlapInt2C S11int(primary_,primary_);
    OverlapInt2C S12int(primary_,minao_);
    OverlapInt2C S22int(minao_,minao_);

    for (size_t P2 = 0; P2 < primary_shells.size(); P2++) {
    for (size_t Q2 = 0; Q2 < primary_shells.size(); Q2++) {
        size_t P = primary_shells[P2];
        size_t Q = primary_shells[Q2];
        int nP = primary_->shell(P).nfunction();
        int nQ = primary_->shell(Q).nfunction();
        int oP = primary_->shell(P).function_index() - primary_->shell(primary_shells[0]).function_index();
        int oQ = primary_->shell(Q).function_index() - primary_->shell(primary_shells[0]).function_index();
        S11int.compute_shell(P,Q);
        double* buffer = S11int.buffer();
        for (int p = 0; p < nP; p++) {
        for (int q = 0; q < nQ; q++) {
            S11p[(p + oP) * nbf + (q + oQ)] = (*buffer++);
        }}
    }} 

    for (size_t P2 = 0; P2 < primary_shells.size(); P2++) {
    for (size_t Q2 = 0; Q2 < minao_shells.size(); Q2++) {
        size_t P = primary_shells[P2];
        size_t Q = minao_shells[Q2];
        int nP = primary_->shell(P).nfunction();
        int nQ = minao_->shell(Q).nfunction();
        int oP = primary_->shell(P).function_index() - primary_->shell(primary_shells[0]).function_index();
        int oQ = minao_->shell(Q).function_index() - minao_->shell(minao_shells[0]).function_index();
        S12int.compute_shell(P,Q);
        double* buffer = S12int.buffer();
        for (int p = 0; p < nP; p++) {
        for (int q = 0; q < nQ; q++) {
            S12p[(p + oP) * nmin + (q + oQ)] = (*buffer++);
        }}
    }} 

    for (size_t P2 = 0; P2 < minao_shells.size(); P2++) {
    for (size_t Q2 = 0; Q2 < minao_shells.size(); Q2++) {
        size_t P = minao_shells[P2];
        size_t Q = minao_shells[Q2];
        int nP = minao_->shell(P).nfunction();
        int nQ = minao_->shell(Q).nfunction();
        int oP = minao_->shell(P).function_index() - minao_->shell(minao_shells[0]).function_index();
        int oQ = minao_->shell(Q).function_index() - minao_->shell(minao_shells[0]).function_index();
        S22int.compute_shell(P,Q);
        double* buffer = S22int.buffer();
        for (int p = 0; p < nP; p++) {
        for (int q = 0; q < nQ; q++) {
            S22p[(p + oP) * nmin + (q + oQ)] = (*buffer++);
        }}
    }} 

    for (int p = 0; p < nmin; p++) {
        S22p[p*nmin + p] -= 1.0;
    }

    double norm = S22.norm(0);
    if (norm > 1.0E-4) throw std::runtime_error("SAD: MinAO basis is not atom-block-orthonormal.");

    Tensor S11inv = S11.power(-1.0,1.0E-12);
    Tensor V12 = S12.clone(kCore);
    V12("pi") = S12("pi") * f("i");
    Tensor C   = Tensor::build(kCore,"C",{nbf,nmin});
    C("pi") = S11inv("pq") * V12("qi");
    
    return C;
} 

} // namespace lightspeed
