set(LIBLIGHTSPEED_HEADERS
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
    df/df.h
    jk/jk.h
    solver/diis.h
    sad/sad.h
)

set(LIBLIGHTSPEED_SOURCES
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
    df/df.cc
    df/aodf.cc
    df/modf.cc
    jk/jk.cc
    jk/directjk.cc
    jk/dfjk.cc
    solver/diis.cc
    sad/sad.cc
)

# build static libray
add_library(lightspeed STATIC ${LIBLIGHTSPEED_SOURCES} ${LIBLIGHTSPEED_HEADERS})
#add_library(lightspeed SHARED ${LIBLIGHTSPEED_SOURCES} ${LIBLIGHTSPEED_HEADERS})

#target_link_libraries(lightspeed ${LIBINT}/lib/libint.so ${LIBAMBIT}/lib/libambit.a)
target_link_libraries(lightspeed ${LIBINT}/lib/libint2.a ${LIBAMBIT}/lib/libambit.a ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})

#install(TARGETS lightspeed ARCHIVE DESTINATION lib)
#install(TARGETS lightspeed SHARED DESTINATION lib)

