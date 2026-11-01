cmake_minimum_required( VERSION 3.16.3 )

project( arx5-sdk )

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON) # global PIC

find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(orocos_kdl REQUIRED)
find_package(kdl_parser REQUIRED)
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)
find_package(Boost REQUIRED COMPONENTS container)

add_compile_definitions(SDK_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm.*|arm|aarch64")
    set(LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/aarch64)

elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64)|(^i686)")
    set(LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/x86_64)
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${kdl_parser_INCLUDE_DIRS})
if(DEFINED ENV{CONDA_PREFIX} AND NOT "$ENV{CONDA_PREFIX}" STREQUAL "")
    include_directories($ENV{CONDA_PREFIX}/include/ $ENV{CONDA_PREFIX}/include/urdfdom_headers)
endif()

add_library(ArxJointController SHARED
    src/app/joint_controller.cpp
    src/app/controller_base.cpp
    src/utils.cpp
)
target_link_libraries(ArxJointController
    ${LIB_DIR}/libhardware.so
    ${LIB_DIR}/libsolver.so
    ${LIB_DIR}/libsoem.so
    Eigen3::Eigen
    Threads::Threads
    spdlog::spdlog
    kdl_parser::kdl_parser
    orocos-kdl
)
add_library(ArxCartesianController SHARED
    src/app/cartesian_controller.cpp
    src/app/controller_base.cpp
    src/utils.cpp
)
target_link_libraries(ArxCartesianController
    ${LIB_DIR}/libhardware.so
    ${LIB_DIR}/libsolver.so
    ${LIB_DIR}/libsoem.so
    Eigen3::Eigen
    Threads::Threads
    spdlog::spdlog
    kdl_parser::kdl_parser
    orocos-kdl
)

# Conda and system ROS installations expose kdl_parser through different paths.
if(DEFINED ENV{CONDA_PREFIX} AND NOT "$ENV{CONDA_PREFIX}" STREQUAL "")
    target_include_directories(ArxJointController PUBLIC $ENV{CONDA_PREFIX}/include/kdl_parser $ENV{CONDA_PREFIX}/include/urdfdom_headers)
    target_include_directories(ArxCartesianController PUBLIC $ENV{CONDA_PREFIX}/include/kdl_parser $ENV{CONDA_PREFIX}/include/urdfdom_headers)
endif()


add_executable(test_joint_controller examples/test_joint_controller.cpp)
target_link_libraries(test_joint_controller 
    ${LIB_DIR}/libhardware.so 
    ${LIB_DIR}/libsolver.so 
    ${LIB_DIR}/libsoem.so
    ArxJointController
    Eigen3::Eigen
    Threads::Threads
    spdlog::spdlog
    kdl_parser::kdl_parser
    orocos-kdl
)

add_executable(test_cartesian_controller examples/test_cartesian_controller.cpp)
target_link_libraries(test_cartesian_controller 
    ${LIB_DIR}/libhardware.so 
    ${LIB_DIR}/libsolver.so
    ${LIB_DIR}/libsoem.so
    ArxCartesianController
    spdlog::spdlog
    Eigen3::Eigen
    Threads::Threads
    kdl_parser::kdl_parser
    orocos-kdl
)

add_subdirectory(python)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}
)

install(FILES ${LIB_DIR}/libhardware.so ${LIB_DIR}/libsolver.so ${LIB_DIR}/libsoem.so
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/${PROJECT_NAME}
)

install(TARGETS ArxJointController ArxCartesianController
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/${PROJECT_NAME}
)
