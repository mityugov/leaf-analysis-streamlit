#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

py::dict find_symmetry(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> mask,
    double lambda_,
    int start_x,
    int start_y);
