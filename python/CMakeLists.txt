set(LIBPYGAUSSIAN_HEADERS
)

set(LIBPYGAUSSIAN_SOURCES
    libpygaussian.cc
    export_collections.cc
    export_tensor.cc
    export_core.cc
    export_mints.cc
    export_ob.cc
)

add_library(pygaussian SHARED ${LIBPYGAUSSIAN_SOURCES} ${LIBPYGAUSSIAN_HEADERS})

target_link_libraries(pygaussian gaussian -lboost_python -lboost_system)

#install(TARGETS pygaussian SHARED DESTINATION lib)

