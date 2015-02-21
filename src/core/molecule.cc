#include <math.h>
#include "molecule.h"

namespace libgaussian {

double SAtom::distance(const SAtom& other) const 
{
    double dx = x_ - other.x_;
    double dy = y_ - other.y_;
    double dz = z_ - other.z_;
    return sqrt(dx*dx + dy*dy + dz*dz);
}

SMolecule::SMolecule(
    const std::string& name,
    const std::vector<SAtom>& atoms) :
    name_(name),
    atoms_(atoms)
{
    for (size_t A = 0; A < atoms_.size(); A++) {
        atoms_[A].set_index(A);
    }
}
void SMolecule::print(
    FILE* fh,
    bool angstrom) const
{
    // PSI4's bohr2ang
    double constexpr bohr2ang = 0.52917720859;
    double factor = (angstrom ? bohr2ang : 1.0);

    fprintf(fh,"  SMolecule: %s\n", name_.c_str());
    fprintf(fh,"    Natom = %zu\n\n", natom());

    fprintf(fh,"    %-12s %18s %18s %18s\n", 
        "Atom", "X", "Y", "Z");
    for (size_t A = 0; A < natom(); A++) {
        const SAtom& atom = atoms_[A];
        fprintf(fh,"    %-12s %18.12f %18.12f %18.12f\n", 
            atom.label().c_str(),
            factor*atom.x(),
            factor*atom.y(),
            factor*atom.z());
    }   
    fprintf(fh,"    Printed in %s\n", (angstrom ? "Angstrom" : "Bohr"));
    fprintf(fh, "\n");
}
double SMolecule::nuclear_repulsion_energy(
    bool use_nuclear,
    double a,
    double b,
    double w) const
{
    double E = 0.0;
    for (size_t A = 0; A < atoms_.size(); A++) {
        for (size_t B = A + 1; B < atoms_.size(); B++) {
            double ZAB = (use_nuclear ?
                atoms_[A].Z() * atoms_[B].Z() :
                atoms_[A].Q() * atoms_[B].Q());
            double rAB = atoms_[A].distance(atoms_[B]);
            E += ZAB * (a / rAB + b * erf(w * rAB) / rAB);
        }
    }
    return E;
}
double SMolecule::nuclear_repulsion_energy_other(
    const std::shared_ptr<SMolecule>& other,
    bool use_nuclear_this,
    bool use_nuclear_other,
    double a,
    double b,
    double w) const
{
    const std::vector<SAtom>& atomsB = other->atoms(); 

    double E = 0.0;
    for (size_t A = 0; A < atoms_.size(); A++) {
        for (size_t B = 0; B < atomsB.size(); B++) {
            double ZA = (use_nuclear_this  ? atoms_[A].Z() : atoms_[A].Q());
            double ZB = (use_nuclear_other ? atomsB[B].Z() : atomsB[B].Q());
            double ZAB = ZA * ZB;
            if (ZAB != 0.0) {
                double rAB = atoms_[A].distance(atomsB[B]);
                E += ZAB * (a / rAB + b * erf(w * rAB) / rAB);
            }
        }
    }
    return E;
}

} // namespace libgaussian
