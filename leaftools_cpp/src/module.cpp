#include <pybind11/pybind11.h>
#include <opencv2/core.hpp>
#include <string>

#include "rotation_profile/rotation_profile_wrap.h"
#include "petiole/petiole_wrap.h"
#include "symmetry/symmetry_wrap.h"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "leaftools: дескриптор формы, отрезание черешка, симметрия листа";

    m.def("cv_version", []() { return std::string(CV_VERSION); },
          "Версия OpenCV, с которой собран модуль.");

    m.def("generate_descriptor", &generate_descriptor,
          py::arg("mask"), py::arg("cx"), py::arg("cy"), py::arg("area"),
          "Дескриптор формы листа (профиль вращения).\n"
          "mask: бинарная маска (H,W) uint8; cx,cy: центр масс; area: площадь.\n"
          "Возвращает dict: angles, jaccard_values, count.");

    m.def("find_petiole_points", &find_petiole_points,
          py::arg("mask"),
          py::arg("svm_model_path") = "",
          py::arg("svm_csv_path") = "",
          "Поиск точек основания черешка (A, B, S) на маске листа.\n"
          "mask: бинарная маска (H,W) uint8, лист=255 на чёрном.\n"
          "Возвращает dict: found, ax,ay, bx,by, sx,sy.");

    m.def("find_symmetry", &find_symmetry,
          py::arg("mask"),
          py::arg("lambda_") = 0.5,
          py::arg("start_x") = -1,
          py::arg("start_y") = -1,
          "Поиск криволинейной оси симметрии листа.\n"
          "mask: бинарная маска (H,W) uint8, лист=255 на чёрном.\n"
          "lambda_: параметр сглаживания оси, 0..1.\n"
          "start_x, start_y: начальная точка (-1 = искать автоматически).\n"
          "Возвращает dict: found, jaccard_straightened, jaccard_original,\n"
          "axis, straightened, reflected.");
}