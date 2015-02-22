#include <memory>
#include <boost/python.hpp>
#include <boost/python/overloads.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <tensor/tensor.h>

using namespace tensor;
using namespace boost::python;

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(tensor_print_ov, Tensor::print, 0, 4)

void export_tensor()
{
    enum_<TensorType>("TensorType")
        .value("kCurrent", kCurrent)
        .value("kCore", kCore)
        .value("kDisk", kDisk)
        .value("kDistributed", kDistributed)
        .value("kDisk", kDisk)
        ;

    class_<Dimension>("Dimension")
        .def(vector_indexing_suite<Dimension>())
        ;

    class_<Tensor>("Tensor", no_init)
        .def("build", &Tensor::build)
        .staticmethod("build")
        .def("printf", &Tensor::print,tensor_print_ov())
        ;     
}
