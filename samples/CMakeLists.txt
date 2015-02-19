set(EXAMPLE2_SOURCES
    example2.cc
)

# build static libray
add_executable(example2 ${EXAMPLE2_SOURCES})
add_dependencies(example2 gaussian)
target_link_libraries(example2 gaussian ${Boost_LIBRARIES})

