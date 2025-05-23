cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 17)
project(cgal_wrapper)

# Find TBB package
set(BOOST_ROOT /opt/boost)
find_package(Threads REQUIRED)
find_package(Boost REQUIRED)

# Find CGAL package
set(CGAL_DIR ../external/cgal/build)
find_package(CGAL REQUIRED)

# Add your source files
set(SOURCE_FILES
    ops.cpp
    )

# Add header files
set(HEADER_FILES
    ops.h
    )

# Set the -fPIC flag for all targets in this CMakeLists.txt
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

# Create a static library
add_library(cgal_wrapper STATIC ${SOURCE_FILES} ${HEADER_FILES})

# Link CGAL and CGAL dependencies
target_link_libraries(cgal_wrapper  PUBLIC Threads::Threads)
target_compile_options(cgal_wrapper PUBLIC -DNOMINMAX)

# Set include directories
target_include_directories(cgal_wrapper PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CGAL_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIR}
)

# Install the library
install(TARGETS cgal_wrapper
    ARCHIVE DESTINATION lib
)

# Install the header files
install(FILES ${HEADER_FILES}
    DESTINATION include/MyCGALLibrary
)
