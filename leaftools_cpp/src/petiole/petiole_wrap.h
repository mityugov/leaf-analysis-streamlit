#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <string>

namespace py = pybind11;

py::dict find_petiole_points(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> mask,
    const std::string& svm_model_path,
    const std::string& svm_csv_path);
