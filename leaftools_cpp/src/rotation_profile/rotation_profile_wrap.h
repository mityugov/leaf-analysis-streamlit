#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

py::dict generate_descriptor(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> mask,
    int cx, int cy, int area);
