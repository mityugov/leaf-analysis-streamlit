#include "rotation_profile_wrap.h"

#include <opencv2/core.hpp>
#include <stdexcept>
#include <map>

#include "RotationProfile.h"

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
