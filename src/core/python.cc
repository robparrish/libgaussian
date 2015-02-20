#include <boost/python.hpp>
#include "am.h"

using namespace libgaussian;
using namespace boost::python;

BOOST_PYTHON_MODULE(core)
{
    class_<SAngularMomentum>("SAngularMomentum", init<int>())
        .def("build", &SAngularMomentum::build)
        .staticmethod("build")
        .add_property("am", &SAngularMomentum::am)
        .add_property("ncartesian", &SAngularMomentum::ncartesian)
        .add_property("nspherical", &SAngularMomentum::nspherical)
        //.add_property("ls", &SAngularMomentum::ls)
        //.add_property("ms", &SAngularMomentum::ms)
        //.add_property("ns", &SAngularMomentum::ns)
        //.add_property("ncoef", &SAngularMomentum::ncoef)
        //.add_property("cartesian_inds", &SAngularMomentum::cartesian_inds)
        //.add_property("spherical_inds", &SAngularMomentum::spherical_inds)
        //.add_property("cartesian_coefs", &SAngularMomentum::cartesian_coefs);
        .def("l", &SAngularMomentum::l)
        .def("m", &SAngularMomentum::m)
        .def("n", &SAngularMomentum::n)
        .def("cartesian_inds", &SAngularMomentum::cartesian_ind)
        .def("spherical_inds", &SAngularMomentum::spherical_ind)
        .def("cartesian_coefs", &SAngularMomentum::cartesian_coef);
}
