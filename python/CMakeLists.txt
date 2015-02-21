set(LIBPYGAUSSIAN_HEADERS
)

set(LIBPYGAUSSIAN_SOURCES
    core.cc
)

add_library(pygaussian SHARED ${LIBPYGAUSSIAN_SOURCES} ${LIBPYGAUSSIAN_HEADERS})

target_link_libraries(pygaussian gaussian -lboost_python -lboost_system)

#install(TARGETS pygaussian SHARED DESTINATION lib)

