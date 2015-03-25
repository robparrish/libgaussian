#include <memory>
#include <boost/python.hpp>
#include <boost/python/overloads.hpp>
#include <mints/int2c.h>
#include <mints/int4c.h>
#include <mints/schwarz.h>

using namespace lightspeed;
using namespace boost::python;

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(potential2c_nuc_overloads, PotentialInt2C::set_nuclear_potential, 1, 2)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(schwarz_print_overloads, SchwarzSieve::print, 0, 1)

void export_mints()
{
    class_<fundamentals::parameters::CorrelationFactor>("CorrelationFactor")
        .def(init<const std::vector<double>&, const std::vector<double>&>())
        .def("slater_exponent", &fundamentals::parameters::CorrelationFactor::slater_exponent)
        .def("set_params", &fundamentals::parameters::CorrelationFactor::set_params)
        .def("exponent", &fundamentals::parameters::CorrelationFactor::exponent, return_value_policy<reference_existing_object>())
        .def("coeff", &fundamentals::parameters::CorrelationFactor::coeff, return_value_policy<reference_existing_object>());

    class_<fundamentals::parameters::FittedSlaterCorrelationFactor, bases<fundamentals::parameters::CorrelationFactor>>("FittedSlaterCorrelationFactor", init<const double&>());

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

    class_<base::Int4C, std::shared_ptr<base::Int4C>>("Int4C", init<
        const std::shared_ptr<SBasisSet>&,
        const std::shared_ptr<SBasisSet>&,
        const std::shared_ptr<SBasisSet>&,
        const std::shared_ptr<SBasisSet>&,
        optional<
        int
        >>())
        .def("basis1", &base::Int4C::basis1, return_value_policy<reference_existing_object>())
        .def("basis2", &base::Int4C::basis2, return_value_policy<reference_existing_object>())
        .def("basis3", &base::Int4C::basis3, return_value_policy<reference_existing_object>())
        .def("basis4", &base::Int4C::basis4, return_value_policy<reference_existing_object>())
        .def("deriv", &base::Int4C::deriv)
        .def("data", &base::Int4C::data, return_value_policy<reference_existing_object>())
        .def("is_spherical", &base::Int4C::is_spherical)
        .def("max_am", &base::Int4C::max_am)
        .def("total_am", &base::Int4C::total_am)
        .def("chunk_size", &base::Int4C::chunk_size)
        .def("set_is_spherical", &base::Int4C::set_is_spherical)
        .def("compute_shell", &base::Int4C::compute_shell)
        .def("compute_shell1", &base::Int4C::compute_shell1)
        .def("compute_shell2", &base::Int4C::compute_shell2)
        .def("compute_quartet", &base::Int4C::compute_quartet)
        .def("compute_quartet1", &base::Int4C::compute_quartet1)
        .def("compute_quartet2", &base::Int4C::compute_quartet2)
        ;

    class_<GeneralInt4C, bases<base::Int4C>>("GeneralInt4C", no_init);

    class_<F12Int4C, bases<GeneralInt4C>>("F12Int4C", init<const std::shared_ptr<SBasisSet>&,
                                                           const std::shared_ptr<SBasisSet>&,
                                                           const std::shared_ptr<SBasisSet>&,
                                                           const std::shared_ptr<SBasisSet>&,
                                                           const fundamentals::parameters::CorrelationFactor&,
                                                           optional<int>>());

    class_<F12ScaledInt4C, bases<GeneralInt4C>>("F12ScaledInt4C", init<const std::shared_ptr<SBasisSet>&,
                                                           const std::shared_ptr<SBasisSet>&,
                                                           const std::shared_ptr<SBasisSet>&,
                                                           const std::shared_ptr<SBasisSet>&,
                                                           const fundamentals::parameters::CorrelationFactor&,
                                                           optional<int>>());

    class_<F12SquaredInt4C, bases<GeneralInt4C>>("F12SquaredInt4C", init<const std::shared_ptr<SBasisSet>&,
                                                           const std::shared_ptr<SBasisSet>&,
                                                           const std::shared_ptr<SBasisSet>&,
                                                           const std::shared_ptr<SBasisSet>&,
                                                           const fundamentals::parameters::CorrelationFactor&,
                                                           optional<int>>());

    class_<F12G12Int4C, bases<GeneralInt4C>>("F12G12Int4C", init<const std::shared_ptr<SBasisSet>&,
                                                           const std::shared_ptr<SBasisSet>&,
                                                           const std::shared_ptr<SBasisSet>&,
                                                           const std::shared_ptr<SBasisSet>&,
                                                           const fundamentals::parameters::CorrelationFactor&,
                                                           optional<int>>());

    class_<F12DoubleCommutatorInt4C, bases<GeneralInt4C>>("F12DoubleCommutatorInt4C", init<const std::shared_ptr<SBasisSet>&,
                                                           const std::shared_ptr<SBasisSet>&,
                                                           const std::shared_ptr<SBasisSet>&,
                                                           const std::shared_ptr<SBasisSet>&,
                                                           const fundamentals::parameters::CorrelationFactor&,
                                                           optional<int>>());

    class_<PotentialInt4C, std::shared_ptr<PotentialInt4C>, bases<base::Int4C>>("PotentialInt4C", init<
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
        .def("is_symmetric", &SchwarzSieve::is_symmetric)
        .def("basis1", &SchwarzSieve::basis1, return_value_policy<reference_existing_object>())
        .def("basis2", &SchwarzSieve::basis2, return_value_policy<reference_existing_object>())
        .def("cutoff", &SchwarzSieve::cutoff)
        .def("a", &SchwarzSieve::a)
        .def("b", &SchwarzSieve::b)
        .def("w", &SchwarzSieve::w)
        .def("printf", &SchwarzSieve::print, schwarz_print_overloads())
        .def("shell_pairs", &SchwarzSieve::shell_pairs, return_value_policy<reference_existing_object>())
        .def("shell_estimate_PQPQ", &SchwarzSieve::shell_estimate_PQPQ)
        .def("shell_estimate_PQRS", &SchwarzSieve::shell_estimate_PQRS)
        ;

}
