#include <memory>
#include <boost/python.hpp>
#include <boost/python/overloads.hpp>
#include <df/df.h>

using namespace lightspeed;
using namespace boost::python;

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(df_power_ov, DFERI::metric_power_core, 0, 2)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(aodf_core_ov, AODFERI::compute_ao_task_core, 0, 1)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(aodf_disk_ov, AODFERI::compute_ao_task_disk, 0, 1)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(modf_add_ov, MODFERI::add_mo_task, 3, 5)

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

    class_<AODFERI, std::shared_ptr<AODFERI>, bases<DFERI>>("AODFERI", init<
        const std::shared_ptr<SchwarzSieve>&,
        const std::shared_ptr<SBasisSet>&
        >())
        .def("compute_ao_task_core", &AODFERI::compute_ao_task_core, aodf_core_ov())
        .def("compute_ao_task_disk", &AODFERI::compute_ao_task_disk, aodf_disk_ov())
        ;

    class_<MODFERI, std::shared_ptr<MODFERI>, bases<DFERI>>("MODFERI", init<
        const std::shared_ptr<SchwarzSieve>&,
        const std::shared_ptr<SBasisSet>&
        >())
        .def("clear", &MODFERI::clear)
        .def("add_mo_task", &MODFERI::add_mo_task, modf_add_ov())
        .def("compute_mo_tasks_core", &MODFERI::compute_mo_tasks_core)
        .def("compute_mo_tasks_disk", &MODFERI::compute_mo_tasks_disk)
        ;

}
