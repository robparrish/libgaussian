set(LIBPYLIGHTSPEED_HEADERS
)

set(LIBPYLIGHTSPEED_SOURCES
    python.cc
    export_collections.cc
    export_tensor.cc
    export_core.cc
    export_mints.cc
    export_ob.cc
    export_df.cc
    export_jk.cc
    export_sad.cc
    export_solver.cc
)

add_library(pylightspeed SHARED ${LIBPYLIGHTSPEED_SOURCES} ${LIBPYLIGHTSPEED_HEADERS})

target_link_libraries(pylightspeed
    lightspeed
    ${LIBINT}/lib/libint.a
    ${LIBAMBIT}/lib/libambit.a
    ${LAPACK_LIBRARIES}
    ${BLAS_LIBRARIES}
    ${Boost_LIBRARIES}
    ${PYTHON_LIBRARIES}
)

set_target_properties(pylightspeed PROPERTIES PREFIX "")
set_target_properties(pylightspeed PROPERTIES SUFFIX ${PYTHON_MODULE_EXTENSION})

#install(TARGETS pylightspeed SHARED DESTINATION lib)

