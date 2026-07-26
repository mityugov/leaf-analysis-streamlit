#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <cstring>

#include "rotation_profile.h"

namespace py = pybind11;

// Версия OpenCV, с которой собран модуль
std::string cv_version() {
    return CV_VERSION;
}

// Дескриптор формы - профиль вращения.
//   mask   — бинарная маска (H, W) uint8
//   cx, cy — центр масс листа
//   area   — площадь листа в пикселях
// Возвращает dict: angles, jaccard_values, count.
py::dict generate_descriptor(
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> mask,
        int cx, int cy, int area) {
    py::buffer_info buf = mask.request();

    if (buf.ndim != 2) {
        throw std::runtime_error("Input mask must be a 2D numpy array (H, W)");
    }
    if (area <= 0) {
        throw std::runtime_error("area must be > 0");
    }

    int rows = static_cast<int>(buf.shape[0]);
    int cols = static_cast<int>(buf.shape[1]);

    cv::Mat m(rows, cols, CV_8UC1, buf.ptr);

    std::map<double, double> profile =
        generateDescriptorSequence(m, cv::Point(cx, cy), area);

    const size_t n = profile.size();
    auto angles = py::array_t<double>(n);
    auto values = py::array_t<double>(n);
    double* pa = static_cast<double*>(angles.request().ptr);
    double* pv = static_cast<double*>(values.request().ptr);

    size_t i = 0;
    for (const auto& kv : profile) {
        pa[i] = kv.first;
        pv[i] = kv.second;
        ++i;
    }

    py::dict result;
    result["angles"] = angles;
    result["jaccard_values"] = values;
    result["count"] = static_cast<int>(n);
    return result;
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "Дескриптор формы - профиль вращения";

    m.def("cv_version", &cv_version, "Версия OpenCV, с которой собран модуль");

    m.def("generate_descriptor", &generate_descriptor,
          py::arg("mask"), py::arg("cx"), py::arg("cy"), py::arg("area"),
          "Дескриптор формы листа.\n"
          "mask: бинарная маска (H,W) uint8; cx,cy: центр масс; area: площадь.\n"
          "Возвращает dict: angles, jaccard_values, count.");
}
