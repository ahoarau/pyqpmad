cmake_minimum_required(VERSION 3.22)
project(pyqpmad)

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)

include(FetchContent)
FetchContent_Declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind.git
    GIT_TAG v2.12.0
)
FetchContent_MakeAvailable(nanobind)

FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://github.com/eigen-mirror/eigen.git
    GIT_TAG 5.0.1
)
FetchContent_MakeAvailable(eigen)

FetchContent_Declare(
    qpmad
    GIT_REPOSITORY https://github.com/asherikov/qpmad.git
    GIT_TAG 1.4.0
    SOURCE_SUBDIR
    download_only
)
FetchContent_MakeAvailable(qpmad)
file(
    GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/generated/include/qpmad/config.h
    CONTENT
        "#pragma once
// #define QPMAD_ENABLE_TRACING
// #define QPMAD_USE_HOUSEHOLDER
// #define QPMAD_PEDANTIC_LICENSE
"
)
add_library(qpmad INTERFACE IMPORTED)
target_include_directories(
    qpmad
    INTERFACE
        ${qpmad_SOURCE_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR}/generated/include
        ${CMAKE_CURRENT_BINARY_DIR}/generated/include/qpmad
)
target_link_libraries(qpmad INTERFACE Eigen3::Eigen)

nanobind_add_module(pyqpmad 
    NB_STATIC STABLE_ABI LTO
    qpmad_pywrap.cpp
)
nanobind_add_stub(
    pyqpmad_stub
    MODULE pyqpmad
    OUTPUT pyqpmad.pyi
    PYTHON_PATH $<TARGET_FILE_DIR:pyqpmad>
    DEPENDS pyqpmad
)

target_link_libraries(pyqpmad PRIVATE qpmad)

install(TARGETS pyqpmad DESTINATION .)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/pyqpmad.pyi DESTINATION .)
