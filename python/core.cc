#include <memory>
#include <boost/python.hpp>
#include <boost/python/overloads.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <core/am.h>
#include <core/molecule.h>
#include <core/basisset.h>

using namespace libgaussian;
using namespace boost::python;

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(smolecule_print_overloads, SMolecule::print, 0, 2)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(smolecule_nuc_overloads, SMolecule::nuclear_repulsion_energy, 0, 4)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(smolecule_nuc_other_overloads, SMolecule::nuclear_repulsion_energy_other, 1, 5)

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(sbasisset_print_overloads, SBasisSet::print, 0, 1)

BOOST_PYTHON_MODULE(libpygaussian)
{
    class_<std::vector<double>>("DoubleVec")
        .def(vector_indexing_suite<std::vector<double>>())
        ;

    class_<std::vector<int>>("IntVec")
        .def(vector_indexing_suite<std::vector<int>>())
        ;

    class_<std::vector<size_t>>("SizetVec")
        .def(vector_indexing_suite<std::vector<size_t>>())
        ;

    class_<SAtom, std::shared_ptr<SAtom>>("SAtom", init<
        const std::string&,
        const std::string&,
        int,
        double,
        double,
        double,
        double,
        double,
        double,
        optional<
        size_t
        >>())
        .def("label", &SAtom::label,   return_value_policy<copy_const_reference>())
        .def("symbol", &SAtom::symbol, return_value_policy<copy_const_reference>())
        .def("N", &SAtom::N)
        .def("x", &SAtom::x)
        .def("y", &SAtom::y)
        .def("z", &SAtom::z)
        .def("Z", &SAtom::Z)
        .def("Ya", &SAtom::Ya)
        .def("Yb", &SAtom::Yb)
        .def("Q", &SAtom::Q)
        .def("S", &SAtom::S)
        .def("index", &SAtom::index)
        .def("set_label", &SAtom::set_label)
        .def("set_symbol", &SAtom::set_symbol)
        .def("set_N", &SAtom::set_N)
        .def("set_x", &SAtom::set_x)
        .def("set_y", &SAtom::set_y)
        .def("set_z", &SAtom::set_z)
        .def("set_Z", &SAtom::set_Z)
        .def("set_Ya", &SAtom::set_Ya)
        .def("set_Yb", &SAtom::set_Yb)
        .def("set_index", &SAtom::set_index)
        .def("distance", &SAtom::distance)
        ;

    class_<std::vector<SAtom>>("SAtomVec")
        .def(vector_indexing_suite<std::vector<SAtom>>())
        ;

    class_<SMolecule, std::shared_ptr<SMolecule>>("SMolecule", init<
        const std::string&,
        const std::vector<SAtom>&
        >())
        .def("name", &SMolecule::name, return_value_policy<copy_const_reference>())
        .def("natom", &SMolecule::natom)
        .def("atom", &SMolecule::atom, return_value_policy<reference_existing_object>())
        .def("atoms", &SMolecule::atoms, return_value_policy<reference_existing_object>())
        .def("printf", &SMolecule::print, smolecule_print_overloads())
        .def("nuclear_repulsion_energy", &SMolecule::nuclear_repulsion_energy, smolecule_nuc_overloads())
        .def("nuclear_repulsion_energy_other", &SMolecule::nuclear_repulsion_energy_other, smolecule_nuc_other_overloads())
        ;

    class_<SAngularMomentum, std::shared_ptr<SAngularMomentum>>("SAngularMomentum", init<int>())
        .def("build", &SAngularMomentum::build)
        .staticmethod("build")
        .def("am", &SAngularMomentum::am)
        .def("ncartesian", &SAngularMomentum::ncartesian)
        .def("nspherical", &SAngularMomentum::nspherical)
        .def("ls", &SAngularMomentum::ls, return_value_policy<reference_existing_object>())
        .def("ms", &SAngularMomentum::ms, return_value_policy<reference_existing_object>())
        .def("ns", &SAngularMomentum::ns, return_value_policy<reference_existing_object>())
        .def("l", &SAngularMomentum::l)
        .def("m", &SAngularMomentum::m)
        .def("n", &SAngularMomentum::n)
        .def("ncoef", &SAngularMomentum::ncoef)
        .def("cartesian_inds", &SAngularMomentum::cartesian_inds,  return_value_policy<reference_existing_object>())
        .def("spherical_inds", &SAngularMomentum::spherical_inds,  return_value_policy<reference_existing_object>())
        .def("cartesian_coefs", &SAngularMomentum::cartesian_coefs,return_value_policy<reference_existing_object>())
        .def("cartesian_ind", &SAngularMomentum::cartesian_ind)
        .def("spherical_ind", &SAngularMomentum::spherical_ind)
        .def("cartesian_coef", &SAngularMomentum::cartesian_coef)
        ;

    class_<std::vector<SAngularMomentum>>("SAngularMomentumVec")
        .def(vector_indexing_suite<std::vector<SAngularMomentum>>())
        ;

    class_<SGaussianShell, std::shared_ptr<SGaussianShell>>("SGaussianShell", init<
        double,
        double,
        double,
        bool,
        int,
        const std::vector<double>&,
        const std::vector<double>&,
        optional<
        size_t,
        size_t,
        size_t,
        size_t
        >>()) 
        .def("x", &SGaussianShell::x)
        .def("y", &SGaussianShell::y)
        .def("z", &SGaussianShell::z)
        .def("is_spherical", &SGaussianShell::is_spherical)
        .def("am", &SGaussianShell::am)
        .def("nfunction", &SGaussianShell::nfunction)
        .def("ncartesian", &SGaussianShell::ncartesian)
        .def("nprimitive", &SGaussianShell::nprimitive)
        .def("c", &SGaussianShell::c)
        .def("e", &SGaussianShell::e)
        .def("cs", &SGaussianShell::cs,return_value_policy<reference_existing_object>()) 
        .def("es", &SGaussianShell::es,return_value_policy<reference_existing_object>()) 
        .def("atom_index", &SGaussianShell::atom_index)
        .def("shell_index", &SGaussianShell::shell_index)
        .def("function_index", &SGaussianShell::function_index)
        .def("cartesian_index", &SGaussianShell::cartesian_index)
        .def("set_x", &SGaussianShell::set_x)
        .def("set_y", &SGaussianShell::set_y)
        .def("set_z", &SGaussianShell::set_z)
        .def("set_is_spherical", &SGaussianShell::set_is_spherical)
        .def("set_am", &SGaussianShell::set_am)
        .def("set_cs", &SGaussianShell::set_cs)
        .def("set_es", &SGaussianShell::set_es)
        .def("set_atom_index", &SGaussianShell::set_atom_index)
        .def("set_shell_index", &SGaussianShell::set_shell_index)
        .def("set_function_index", &SGaussianShell::set_function_index)
        .def("set_cartesian_index", &SGaussianShell::set_cartesian_index)
        ;

    class_<std::vector<SGaussianShell>>("SGaussianShellVec")
        .def(vector_indexing_suite<std::vector<SGaussianShell>>())
        ;

    class_<std::vector<std::vector<SGaussianShell>>>("SGaussianShellVecVec")
        .def(vector_indexing_suite<std::vector<std::vector<SGaussianShell>>>())
        ;

    class_<SBasisSet, std::shared_ptr<SBasisSet>>("SBasisSet", init<
        const std::string&,
        const std::vector<std::vector<SGaussianShell>>&
        >())
        .def("zero_basis", &SBasisSet::zero_basis)
        .staticmethod("zero_basis")
        .def("name", &SBasisSet::name, return_value_policy<copy_const_reference>())
        .def("shell", &SBasisSet::shell, return_value_policy<reference_existing_object>())
        .def("shells", &SBasisSet::shells, return_value_policy<reference_existing_object>())
        .def("atoms_to_shell_inds", &SBasisSet::atoms_to_shell_inds, return_value_policy<reference_existing_object>())
        .def("am_info", &SBasisSet::am_info, return_value_policy<reference_existing_object>())
        .def("natom", &SBasisSet::natom)
        .def("nshell", &SBasisSet::nshell)
        .def("nfunction", &SBasisSet::nfunction)
        .def("ncartesian", &SBasisSet::ncartesian)
        .def("nprimitive", &SBasisSet::nprimitive)
        .def("has_spherical", &SBasisSet::has_spherical)
        .def("natom", &SBasisSet::natom)
        .def("max_am", &SBasisSet::max_am)
        .def("max_nfunction", &SBasisSet::max_nfunction)
        .def("max_ncartesian", &SBasisSet::max_ncartesian)
        .def("max_nprimitive", &SBasisSet::max_nprimitive)
        .def("printf", &SBasisSet::print, sbasisset_print_overloads())
        ;
        
}
