set(LIBGAUSSIAN_HEADERS
    core/molecule.h
    core/basisset.h
    core/am.h
    mints/int2c.h
    mints/int4c.h
    mints/schwarz.h
    mints/constants.h
    mints/fjt.h
    mints/osrecur.h
    ob/ob.h
)

set(LIBGAUSSIAN_SOURCES
    core/molecule.cc
    core/basisset.cc
    core/am.cc
    mints/constants.cc
    mints/fjt.cc
    mints/int2c.cc
    mints/overlap2c.cc
    mints/dipole2c.cc
    mints/quadrupole2c.cc
    mints/kinetic2c.cc
    mints/potential2c.cc
    mints/int4c.cc
    mints/potential4c.cc
    mints/osrecur.cc
    mints/schwarz.cc
    ob/ob.cc
)

# build static libray
#add_library(gaussian STATIC ${LIBGAUSSIAN_SOURCES} ${LIBGAUSSIAN_HEADERS})
add_library(gaussian SHARED ${LIBGAUSSIAN_SOURCES} ${LIBGAUSSIAN_HEADERS})

target_link_libraries(gaussian ${LIBINT}/lib/libint.so ${LIBTENSOR}/lib/libtensor.a)

#install(TARGETS gaussian ARCHIVE DESTINATION lib)
#install(TARGETS gaussian SHARED DESTINATION lib)

