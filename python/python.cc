#include <boost/python.hpp>

using namespace boost::python;

void export_collections();
void export_ambit();
void export_core();
void export_mints();
void export_ob();
void export_df();
void export_jk();
void export_sad();
void export_solver();

BOOST_PYTHON_MODULE(pylightspeed)
{
    export_collections();
    export_ambit();
    export_core();
    export_mints();
    export_ob();
    export_df();
    export_jk();
    export_sad();
    export_solver();
}
