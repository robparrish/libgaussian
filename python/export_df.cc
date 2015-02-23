#include <memory>
#include <boost/python.hpp>
#include <boost/python/overloads.hpp>
#include <df/df.h>

using namespace libgaussian;
using namespace boost::python;

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(df_power_ov, DFERI::metric_power_core, 0, 2)

void export_df()
{
    class_<DFERI, std::shared_ptr<DFERI>>("DFERI", init<
        const std::shared_ptr<SchwarzSieve>&,
        const std::shared_ptr<SBasisSet>&
        >())
        .def("sieve", &DFERI::sieve, return_value_policy<reference_existing_object>())
        .def("primary", &DFERI::primary, return_value_policy<reference_existing_object>())
        .def("auxiliary", &DFERI::auxiliary, return_value_policy<reference_existing_object>())
        .def("doubles", &DFERI::doubles)
        .def("a", &DFERI::a)
        .def("b", &DFERI::b)
        .def("w", &DFERI::w)
        .def("metric_condition", &DFERI::metric_condition)
        .def("set_doubles", &DFERI::set_doubles)
        .def("set_a", &DFERI::set_a)
        .def("set_b", &DFERI::set_b)
        .def("set_w", &DFERI::set_w)
        .def("set_metric_condition", &DFERI::set_metric_condition)
        .def("metric_core", &DFERI::metric_core)
        .def("metric_power_core", &DFERI::metric_power_core, df_power_ov())
        ;
}
