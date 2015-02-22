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

    enum_<EigenvalueOrder>("EigenvalueOrder")
        .value("kAscending", kAscending)
        .value("kDescending", kDescending)
        ;

    class_<Dimension>("Dimension")
        .def(vector_indexing_suite<Dimension>())
        ;

    class_<Dimension>("Indices")
        .def(vector_indexing_suite<Indices>())
        ;

    class_<aligned_vector<double>>("AlignedVec")
        .def(vector_indexing_suite<aligned_vector<double>>())
        ;

    class_<std::vector<Tensor>>("TensorVec")
        .def(vector_indexing_suite<std::vector<Tensor>>())
        ;

    class_<Tensor>("Tensor", no_init)
        .def("build", &Tensor::build)
        .staticmethod("build")
        .def("clone", &Tensor::clone)
        .def("type", &Tensor::type)
        .def("name", &Tensor::name)
        .def("dims", &Tensor::dims, return_value_policy<reference_existing_object>())
        .def("dim", &Tensor::dim)
        .def("rank", &Tensor::rank)
        .def("numel", &Tensor::numel)
        .def("set_name", &Tensor::set_name)
        .def(self == self)
        .def(self != self)
        .def("printf", &Tensor::print,tensor_print_ov())
        //.def("data", &Tensor::data)
        .def("norm", &Tensor::norm)
        .def("zero", &Tensor::zero)
        .def("scale", &Tensor::scale)
        .def("copy", &Tensor::copy)
        .def("permute", &Tensor::permute)
        .def("contract", &Tensor::contract)
        .def("gemm", &Tensor::gemm)
        .def("syev", &Tensor::syev)
        .def("power", &Tensor::power)
        .def("set_scratch_path", &Tensor::set_scratch_path)
        .staticmethod("set_scratch_path")
        .def("scratch_path", &Tensor::scratch_path)
        .staticmethod("scratch_path")
        ;     

}
