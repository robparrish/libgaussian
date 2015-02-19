set(EXAMPLE1_SOURCES
    example1.cc
)

# build static libray
add_executable(example1 ${EXAMPLE1_SOURCES})
add_dependencies(example1 gaussian)
target_link_libraries(example1 gaussian ${Boost_LIBRARIES})

set(EXAMPLE2_SOURCES
    example2.cc
)

# build static libray
add_executable(example2 ${EXAMPLE2_SOURCES})
add_dependencies(example2 gaussian)
target_link_libraries(example2 gaussian ${Boost_LIBRARIES})

