#include <memory>
#include <boost/python.hpp>
#include <boost/python/overloads.hpp>
#include <sad/sad.h>

using namespace lightspeed;
using namespace boost::python;

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(sad_print_ov, SAD::print, 0, 1)

void export_sad()
{
    class_<SAD, std::shared_ptr<SAD>>("SAD", init<
        const std::shared_ptr<SMolecule>&,
        const std::shared_ptr<SBasisSet>&,
        const std::shared_ptr<SBasisSet>&
        >())
        .def("printf", &SAD::print, sad_print_ov())
        .def("compute_C", &SAD::compute_C)
        .def("compute_Ca", &SAD::compute_Ca)
        .def("compute_Cb", &SAD::compute_Cb)
        ;

}
