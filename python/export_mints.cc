#include <memory>
#include <boost/python.hpp>
#include <boost/python/overloads.hpp>
#include <mints/int2c.h>
#include <mints/int4c.h>
#include <mints/schwarz.h>

using namespace libgaussian;
using namespace boost::python;

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(potential2c_nuc_overloads, PotentialInt2C::set_nuclear_potential, 1, 2)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(schwarz_print_overloads, SchwarzSieve::print, 0, 1)

void export_mints()
{
    class_<Int2C, std::shared_ptr<Int2C>>("Int2C", init<
        const std::shared_ptr<SBasisSet>&,
        const std::shared_ptr<SBasisSet>&,
        optional<
        int
        >>())
        .def("basis1", &Int2C::basis1, return_value_policy<reference_existing_object>())
        .def("basis2", &Int2C::basis2, return_value_policy<reference_existing_object>())
        .def("deriv", &Int2C::deriv)
        .def("data", &Int2C::data, return_value_policy<reference_existing_object>())
        .def("is_spherical", &Int2C::is_spherical)
        .def("x", &Int2C::x)
        .def("y", &Int2C::y)
        .def("z", &Int2C::z)
        .def("max_am", &Int2C::max_am)
        .def("total_am", &Int2C::total_am)
        .def("chunk_size", &Int2C::chunk_size)
        .def("set_is_spherical", &Int2C::set_is_spherical)
        .def("set_x", &Int2C::set_x)
        .def("set_y", &Int2C::set_y)
        .def("set_z", &Int2C::set_z)
        .def("compute_shell", &Int2C::compute_shell)
        .def("compute_shell1", &Int2C::compute_shell1)
        .def("compute_shell2", &Int2C::compute_shell2)
        .def("compute_pair", &Int2C::compute_pair)
        .def("compute_pair1", &Int2C::compute_pair1)
        .def("compute_pair2", &Int2C::compute_pair2)
        ;

    class_<OverlapInt2C, std::shared_ptr<OverlapInt2C>, bases<Int2C>>("OverlapInt2C", init<
        const std::shared_ptr<SBasisSet>&,
        const std::shared_ptr<SBasisSet>&,
        optional<
        int
        >>())
        ;

    class_<DipoleInt2C, std::shared_ptr<DipoleInt2C>, bases<Int2C>>("DipoleInt2C", init<
        const std::shared_ptr<SBasisSet>&,
        const std::shared_ptr<SBasisSet>&,
        optional<
        int
        >>())
        ;

    class_<QuadrupoleInt2C, std::shared_ptr<QuadrupoleInt2C>, bases<Int2C>>("QuadrupoleInt2C", init<
        const std::shared_ptr<SBasisSet>&,
        const std::shared_ptr<SBasisSet>&,
        optional<
        int
        >>())
        ;

    class_<KineticInt2C, std::shared_ptr<KineticInt2C>, bases<Int2C>>("KineticInt2C", init<
        const std::shared_ptr<SBasisSet>&,
        const std::shared_ptr<SBasisSet>&,
        optional<
        int
        >>())
        ;

    class_<PotentialInt2C, std::shared_ptr<PotentialInt2C>, bases<Int2C>>("PotentialInt2C", init<
        const std::shared_ptr<SBasisSet>&,
        const std::shared_ptr<SBasisSet>&,
        optional<
        int,
        double,
        double,
        double
        >>())
        .def("a", &PotentialInt2C::a)
        .def("b", &PotentialInt2C::b)
        .def("w", &PotentialInt2C::w)
        .def("xs", &PotentialInt2C::xs, return_value_policy<reference_existing_object>())
        .def("ys", &PotentialInt2C::ys, return_value_policy<reference_existing_object>())
        .def("zs", &PotentialInt2C::zs, return_value_policy<reference_existing_object>())
        .def("Zs", &PotentialInt2C::Zs, return_value_policy<reference_existing_object>())
        .def("set_nuclear_potential", &PotentialInt2C::set_nuclear_potential,potential2c_nuc_overloads()) 
        ;

    class_<Int4C, std::shared_ptr<Int4C>>("Int4C", init<
        const std::shared_ptr<SBasisSet>&,
        const std::shared_ptr<SBasisSet>&,
        const std::shared_ptr<SBasisSet>&,
        const std::shared_ptr<SBasisSet>&,
        optional<
        int
        >>())
        .def("basis1", &Int4C::basis1, return_value_policy<reference_existing_object>())
        .def("basis2", &Int4C::basis2, return_value_policy<reference_existing_object>())
        .def("basis3", &Int4C::basis3, return_value_policy<reference_existing_object>())
        .def("basis4", &Int4C::basis4, return_value_policy<reference_existing_object>())
        .def("deriv", &Int4C::deriv)
        .def("data", &Int4C::data, return_value_policy<reference_existing_object>())
        .def("is_spherical", &Int4C::is_spherical)
        .def("max_am", &Int4C::max_am)
        .def("total_am", &Int4C::total_am)
        .def("chunk_size", &Int4C::chunk_size)
        .def("set_is_spherical", &Int4C::set_is_spherical)
        .def("compute_shell", &Int4C::compute_shell)
        .def("compute_shell1", &Int4C::compute_shell1)
        .def("compute_shell2", &Int4C::compute_shell2)
        .def("compute_quartet", &Int4C::compute_quartet)
        .def("compute_quartet1", &Int4C::compute_quartet1)
        .def("compute_quartet2", &Int4C::compute_quartet2)
        ;

    class_<PotentialInt4C, std::shared_ptr<PotentialInt4C>, bases<Int4C>>("PotentialInt4C", init<
        const std::shared_ptr<SBasisSet>&,
        const std::shared_ptr<SBasisSet>&,
        const std::shared_ptr<SBasisSet>&,
        const std::shared_ptr<SBasisSet>&,
        optional<
        int,
        double,
        double,
        double
        >>())
        .def("a", &PotentialInt4C::a)
        .def("b", &PotentialInt4C::b)
        .def("w", &PotentialInt4C::w)
        ;

    class_<SchwarzSieve, std::shared_ptr<SchwarzSieve>>("SchwarzSieve", init<
        const std::shared_ptr<SBasisSet>&,
        const std::shared_ptr<SBasisSet>&,
        double,
        optional<
        double,
        double,
        double
        >>())
        .def("symmetric", &SchwarzSieve::symmetric)
        .def("basis1", &SchwarzSieve::basis1, return_value_policy<reference_existing_object>())
        .def("basis2", &SchwarzSieve::basis2, return_value_policy<reference_existing_object>())
        .def("cutoff", &SchwarzSieve::cutoff)
        .def("a", &SchwarzSieve::a)
        .def("b", &SchwarzSieve::b)
        .def("w", &SchwarzSieve::w)
        .def("printf", &SchwarzSieve::print, schwarz_print_overloads())
        .def("shell_estimate_PQPQ", &SchwarzSieve::shell_estimate_PQPQ)
        .def("shell_estimate_PQRS", &SchwarzSieve::shell_estimate_PQRS)
        ;

}
