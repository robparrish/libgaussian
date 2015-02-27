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

target_link_libraries(pylightspeed lightspeed -lboost_python -lboost_system)

#install(TARGETS pylightspeed SHARED DESTINATION lib)

