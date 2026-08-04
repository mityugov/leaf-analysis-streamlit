#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <cstring>
#include <string>
#include <map>

#include "rotation_profile.h"
#include "petiole/LeafPetiolePointPairFinder.h"

namespace py = pybind11;

std::string cv_version() {
    return CV_VERSION;
}

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

py::dict find_petiole_points(
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> mask,
        const std::string& svm_model_path,
        const std::string& svm_csv_path) {
    py::buffer_info buf = mask.request();

    if (buf.ndim != 2) {
        throw std::runtime_error("Input mask must be a 2D numpy array (H, W)");
    }

    int rows = static_cast<int>(buf.shape[0]);
    int cols = static_cast<int>(buf.shape[1]);

    // Копируем в отдельный cv::Mat (алгоритм внутри может модифицировать данные).
    cv::Mat m(rows, cols, CV_8UC1, buf.ptr);
    cv::Mat binary = m.clone();

    LeafPetiolePointPairFinder finder(binary, svm_model_path, svm_csv_path);
    FindResult res = finder.findBestPair();

    py::dict result;
    result["found"] = res.found;
    result["ax"] = res.ptA.x;  result["ay"] = res.ptA.y;
    result["bx"] = res.ptB.x;  result["by"] = res.ptB.y;
    result["sx"] = res.ptS.x;  result["sy"] = res.ptS.y;
    return result;
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "Дескриптор формы листа (профиль вращения) на OpenCV";

    m.def("cv_version", &cv_version, "Версия OpenCV, с которой собран модуль");

    m.def("generate_descriptor", &generate_descriptor,
          py::arg("mask"), py::arg("cx"), py::arg("cy"), py::arg("area"),
          "Дескриптор формы листа.\n"
          "mask: бинарная маска (H,W) uint8; cx,cy: центр масс; area: площадь.\n"
          "Возвращает dict: angles, jaccard_values, count.");

    m.def("find_petiole_points", &find_petiole_points,
          py::arg("mask"),
          py::arg("svm_model_path") = "",
          py::arg("svm_csv_path") = "",
          "Поиск точек основания черешка (A, B, S) на маске листа.\n"
          "mask: бинарная маска (H,W) uint8, лист=255 на чёрном.\n"
          "Возвращает dict: found, ax,ay, bx,by, sx,sy.");
}