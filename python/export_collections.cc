#include <memory>
#include <boost/python.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <tensor/tensor.h>
#include <core/am.h>
#include <core/molecule.h>
#include <core/basisset.h>

using namespace libgaussian;
using namespace tensor;
using namespace boost::python;

void export_collections()
{

    class_<std::vector<int>>("IntVec")
        .def(vector_indexing_suite<std::vector<int>>())
        ;

    class_<std::pair<int, int>>("IntPair")
        .def_readwrite("first", &std::pair<int,int>::first)
        .def_readwrite("second", &std::pair<int,int>::second)
        ;

    class_<std::vector<std::pair<int,int>>>("IntPairVec")
        .def(vector_indexing_suite<std::vector<std::pair<int,int>>>())
        ;

    class_<std::vector<size_t>>("Size_tVec")
        .def(vector_indexing_suite<std::vector<size_t>>())
        ;

    class_<std::vector<std::vector<size_t>>>("Size_tVecVec")
        .def(vector_indexing_suite<std::vector<std::vector<size_t>>>())
        ;

    class_<std::vector<double>>("DoubleVec")
        .def(vector_indexing_suite<std::vector<double>>())
        ;

    class_<std::vector<std::string>>("StringVec")
        .def(vector_indexing_suite<std::vector<std::string>>())
        ;

    class_<aligned_vector<double>>("AlignedVec")
        .def(vector_indexing_suite<aligned_vector<double>>())
        ;

    class_<std::vector<Tensor>>("TensorVec")
        .def(vector_indexing_suite<std::vector<Tensor>>())
        ;

    class_<std::map<std::string, Tensor>>("TensorMap")
        .def(map_indexing_suite<std::map<std::string, Tensor>>())
        ;

    class_<std::vector<SAtom>>("SAtomVec")
        .def(vector_indexing_suite<std::vector<SAtom>>())
        ;

    class_<std::vector<SAngularMomentum>>("SAngularMomentumVec")
        .def(vector_indexing_suite<std::vector<SAngularMomentum>>())
        ;

    class_<std::vector<SGaussianShell>>("SGaussianShellVec")
        .def(vector_indexing_suite<std::vector<SGaussianShell>>())
        ;

    class_<std::vector<std::vector<SGaussianShell>>>("SGaussianShellVecVec")
        .def(vector_indexing_suite<std::vector<std::vector<SGaussianShell>>>())
        ;

}
