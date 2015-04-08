set(LIGHTSPEED_HEADERS
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
    tb/tb.h
)

set(LIGHTSPEED_SOURCES
    core/molecule.cc
    core/basisset.cc
    core/am.cc
    mints/constants.cc
    mints/fjt.cc
    mints/general4c.cc
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
    tb/tb.cc
)

# build static libray
add_library(lightspeed SHARED ${LIGHTSPEED_SOURCES} ${LIGHTSPEED_HEADERS})

target_link_libraries(lightspeed ${LIBINT}/lib/libint2.a -L${LIBAMBIT}/lib -lambit ${Boost_LIBRARIES})

#set_target_properties(lightspeed PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
#set_target_properties(lightspeed PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")

install(TARGETS lightspeed LIBRARY DESTINATION lib)
foreach(file ${LIGHTSPEED_HEADERS})
    get_filename_component(dir ${file} DIRECTORY)
    install(FILES ${file} DESTINATION include/lightspeed/${dir})
endforeach()
