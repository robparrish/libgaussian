#include <memory>
#include <boost/python.hpp>
#include <boost/python/overloads.hpp>
#include <jk/jk.h>

using namespace lightspeed;
using namespace boost::python;

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(jk_print_ov, JK::print, 0, 1)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(jk_compute_JK_from_C_ov, JK::compute_JK_from_C, 4, 6)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(jk_compute_JK_from_D_ov, JK::compute_JK_from_D, 4, 6)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(directjk_print_ov, DirectJK::print, 0, 1)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(directjk_compute_JK_from_C_ov, DirectJK::compute_JK_from_C, 4, 6)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(directjk_compute_JK_from_D_ov, DirectJK::compute_JK_from_D, 4, 6)

void export_jk()
{
    enum_<JKType>("JKType")
        .value("kBase", kBase)
        .value("kDirect", kDirect)
        .value("kDF", kDF)
        ;

    class_<JK, std::shared_ptr<JK>>("JK", init<
        const std::shared_ptr<SchwarzSieve>&
        >())
        .def("primary", &JK::primary, return_value_policy<reference_existing_object>())
        .def("sieve", &JK::sieve, return_value_policy<reference_existing_object>())
        .def("doubles", &JK::doubles)
        .def("compute_J", &JK::compute_J)
        .def("compute_K", &JK::compute_K)
        .def("a", &JK::a)
        .def("b", &JK::b)
        .def("w", &JK::w)
        .def("product_cutoff", &JK::product_cutoff)
        .def("set_doubles", &JK::set_doubles)
        .def("set_compute_J", &JK::set_compute_J)
        .def("set_compute_K", &JK::set_compute_K)
        .def("set_a", &JK::set_a)
        .def("set_b", &JK::set_b)
        .def("set_w", &JK::set_w)
        .def("set_product_cutoff", &JK::set_product_cutoff)
        .def("initialize", &JK::initialize)
        .def("printf", &JK::print, jk_print_ov())
        .def("compute_JK_from_C", &JK::compute_JK_from_C, jk_compute_JK_from_C_ov())
        .def("compute_JK_from_D", &JK::compute_JK_from_D, jk_compute_JK_from_D_ov())
        .def("finalize", &JK::finalize)
        ;

    class_<DirectJK, std::shared_ptr<DirectJK>, bases<JK>>("DirectJK", init<
        const std::shared_ptr<SchwarzSieve>&
        >())
        .def("compute_JK_from_C", &DirectJK::compute_JK_from_C, directjk_compute_JK_from_C_ov())
        .def("compute_JK_from_D", &DirectJK::compute_JK_from_D, directjk_compute_JK_from_D_ov())
        .def("printf", &DirectJK::print, directjk_print_ov())
        ;

}
