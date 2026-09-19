#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <stdexcept>
#include <cstring>
#include <string>

#include "CurvilinearSymmetryBase.h"

namespace py = pybind11;

static cv::Mat numpy_to_mat(
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> arr) {
    py::buffer_info buf = arr.request();
    if (buf.ndim != 2)
        throw std::runtime_error("mask must be a 2D numpy array (H, W)");
    int rows = static_cast<int>(buf.shape[0]);
    int cols = static_cast<int>(buf.shape[1]);
    cv::Mat m(rows, cols, CV_8UC1, buf.ptr);
    return m.clone();
}

static py::array mat_to_numpy(const cv::Mat& mat) {
    cv::Mat m = mat.isContinuous() ? mat : mat.clone();
    if (m.channels() == 3) {
        return py::array_t<uint8_t>(
            {m.rows, m.cols, 3},
            {static_cast<size_t>(m.step[0]), static_cast<size_t>(m.step[1]), sizeof(uint8_t)},
            m.ptr<uint8_t>());
    }
    return py::array_t<uint8_t>(
        {m.rows, m.cols},
        {static_cast<size_t>(m.step[0]), sizeof(uint8_t)},
        m.ptr<uint8_t>());
}

py::dict find_symmetry(
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> mask,
        double lambda_,
        int start_x,
        int start_y) {

    if (lambda_ < 0.0 || lambda_ > 1.0)
        throw std::runtime_error("lambda must be in [0, 1]");

    cv::Mat binary = numpy_to_mat(mask);

    Data dt;
    dt.image_ = binary.clone();
    dt.area_ = cv::countNonZero(dt.image_);
    dt.lambda_ = lambda_;
    dt.stepLength = 0;
    dt.out_path_ = "";
    dt.in_path_ = "";

    if (start_x >= 0 && start_y >= 0) {
        dt.start_point_ = cv::Point2f(static_cast<float>(start_x),
                                      static_cast<float>(start_y));
        dt.definedInitialPoint = true;
    } else {
        dt.start_point_ = cv::Point2f(-1, -1);
        dt.definedInitialPoint = false;
    }

    if (dt.takeInfoFromImage() != 0)
        throw std::runtime_error("contour not found in mask");

    GreedySearch tool;
    Curvilinear line;
    tool.processing(dt, line);

    py::dict result;

    bool found = !line.curvilinearAxisAtoms.empty()
                 && !line.curvilinear_axis_.empty();
    result["found"] = found;

    if (!found) {
        result["jaccard_straightened"] = 0.0;
        result["jaccard_original"] = 0.0;
        return result;
    }

    result["jaccard_straightened"] = line.jaccard_index_with_straightened_shape_area_;
    result["jaccard_original"] = line.jaccard_index_with_original_shape_area_;
    result["axis"] = mat_to_numpy(line.curvilinear_axis_);
    result["straightened"] = mat_to_numpy(line.straightened_);
    result["reflected"] = mat_to_numpy(line.reflected_);
    return result;
}
