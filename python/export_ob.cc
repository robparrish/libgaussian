#include <memory>
#include <boost/python.hpp>
#include <boost/python/overloads.hpp>
#include <ob/ob.h>

using namespace lightspeed;
using namespace boost::python;

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(ob_S_overloads, OneBody::compute_S, 1, 2)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(ob_T_overloads, OneBody::compute_T, 1, 2)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(ob_X_overloads, OneBody::compute_X, 1, 3)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(ob_Q_overloads, OneBody::compute_Q, 1, 3)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(ob_V_overloads, OneBody::compute_V, 5, 9)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(ob_V_nuclear_overloads, OneBody::compute_V_nuclear, 2, 7)

void export_ob()
{
    class_<OneBody, std::shared_ptr<OneBody>>("OneBody", init<
        const std::shared_ptr<SchwarzSieve>&
        >())
        .def("is_symmetric", &OneBody::is_symmetric)
        .def("basis1", &OneBody::basis1, return_value_policy<reference_existing_object>())
        .def("basis2", &OneBody::basis2, return_value_policy<reference_existing_object>())
        .def("sieve", &OneBody::sieve, return_value_policy<reference_existing_object>())
        .def("compute_S", &OneBody::compute_S, ob_S_overloads())
        .def("compute_T", &OneBody::compute_T, ob_T_overloads())
        .def("compute_X", &OneBody::compute_X, ob_X_overloads())
        .def("compute_Q", &OneBody::compute_Q, ob_Q_overloads())
        .def("compute_V", &OneBody::compute_V, ob_V_overloads())
        .def("compute_V_nuclear", &OneBody::compute_V_nuclear, ob_V_nuclear_overloads())
        ;

}
