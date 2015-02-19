set(LIBGAUSSIAN_HEADERS
    core/molecule.h
    core/basisset.h
    core/am.h
    mints/int2c.h
    mints/int4c.h
)

set(LIBGAUSSIAN_SOURCES
    core/molecule.cc
    core/basisset.cc
    core/am.cc
    mints/int2c.cc
    mints/overlap2c.cc
    mints/dipole2c.cc
    mints/quadrupole2c.cc
    mints/kinetic2c.cc
    mints/potential2c.cc
    mints/int4c.cc
    mints/potential4c.cc
)

# build static libray
add_library(gaussian STATIC ${LIBGAUSSIAN_SOURCES} ${LIBGAUSSIAN_HEADERS})

target_link_libraries(gaussian ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})

install(TARGETS gaussian ARCHIVE DESTINATION lib)

