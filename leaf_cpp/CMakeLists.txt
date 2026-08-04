cmake_minimum_required(VERSION 3.18)
project(leaftools VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)

pybind11_add_module(_core
    src/module.cpp
    src/rotation_profile.cpp
    src/petiole/LeafPetiolePointPairFinder.cpp
    src/petiole/SVM_Classifier.cpp
)

target_include_directories(_core PRIVATE ${OpenCV_INCLUDE_DIRS} src)
target_link_libraries(_core PRIVATE ${OpenCV_LIBS})

target_compile_definitions(_core PRIVATE VERSION_INFO="${PROJECT_VERSION}")

message(STATUS "OpenCV version: ${OpenCV_VERSION}")

install(TARGETS _core DESTINATION leaftools)