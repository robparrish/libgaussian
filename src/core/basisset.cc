#include <math.h>
#include "basisset.h"

namespace libgaussian {

SBasisSet::SBasisSet(
    const std::string& name,
    const std::vector<std::vector<SGaussianShell>>& shells) :
    name_(name) 
{
    size_t nshell = 0L;
    for (size_t ind1 = 0L; ind1 < shells.size(); ind1++) {
        nshell += shells[ind1].size();
    } 

    atoms_to_shell_inds_.resize(shells.size());
    shells_.resize(nshell);
    size_t shell_ind = 0L;
    size_t function_ind = 0L;
    size_t cartesian_ind = 0L;
    size_t primitive_ind = 0L;
    for (size_t Aind = 0L; Aind < shells.size(); Aind++) {
        atoms_to_shell_inds_[Aind].resize(shells[Aind].size());
        for (size_t Sind = 0L; Sind < shells[Aind].size(); Sind++) {
            atoms_to_shell_inds_[Aind][Sind] = shell_ind;
            shells_[shell_ind] = shells[Aind][Sind];
            shells_[shell_ind].set_atom_index(Aind); 
            shells_[shell_ind].set_shell_index(shell_ind); 
            shells_[shell_ind].set_function_index(function_ind); 
            shells_[shell_ind].set_cartesian_index(cartesian_ind); 
            shell_ind++;
            function_ind += shells_[shell_ind].nfunction();
            cartesian_ind += shells_[shell_ind].ncartesian();
            primitive_ind += shells_[shell_ind].nprimitive();
        }
    }
    nfunction_ = function_ind;
    ncartesian_ = cartesian_ind;
    nprimitive_ = primitive_ind;

    am_info_ = SAngularMomentum::build(max_am());
}
std::shared_ptr<SBasisSet> SBasisSet::zero_basis()
{
    SGaussianShell zero(
        0.0,
        0.0,
        0.0,
        false,
        0,
        {1.0},
        {0.0},
        0,
        0,
        0,
        0);

    std::vector<std::vector<SGaussianShell> > shells = {{zero}};

    return std::shared_ptr<SBasisSet>(new SBasisSet("0", shells));  
}
bool SBasisSet::has_spherical() const 
{
    for (size_t ind = 0; ind < shells_.size(); ind++) {
        if (shells_[ind].is_spherical()) return true;
    }
    return false;
}
int SBasisSet::max_am() const 
{
    int val = 0;
    for (size_t ind = 0; ind < shells_.size(); ind++) {
        val = std::max(val,shells_[ind].am());
    }
    return val;
}
size_t SBasisSet::max_nfunction() const 
{
    size_t val = 0L;
    for (size_t ind = 0; ind < shells_.size(); ind++) {
        val = std::max(val,shells_[ind].nfunction());
    }
    return val;
}
size_t SBasisSet::max_ncartesian() const 
{
    size_t val = 0L;
    for (size_t ind = 0; ind < shells_.size(); ind++) {
        val = std::max(val,shells_[ind].ncartesian());
    }
    return val;
}
size_t SBasisSet::max_nprimitive() const 
{
    size_t val = 0L;
    for (size_t ind = 0; ind < shells_.size(); ind++) {
        val = std::max(val,shells_[ind].nprimitive());
    }
    return val;
}

} // namespace libgaussian
