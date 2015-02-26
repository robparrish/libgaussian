#include <memory>
#include <boost/python.hpp>
#include <boost/python/overloads.hpp>
#include <solver/diis.h>

using namespace lightspeed;
using namespace boost::python;

void export_solver()
{
    class_<DIIS, std::shared_ptr<DIIS>, boost::noncopyable>("DIIS", init<
        size_t,
        size_t,
        bool
        >())
        .def("min_vectors", &DIIS::min_vectors)
        .def("max_vectors", &DIIS::max_vectors)
        .def("cur_vectors", &DIIS::cur_vectors)
        .def("use_disk", &DIIS::use_disk)
        .def("clear", &DIIS::clear)
        .def("add_iteration", &DIIS::add_iteration)
        .def("extrapolate", &DIIS::extrapolate)
        ;

}
