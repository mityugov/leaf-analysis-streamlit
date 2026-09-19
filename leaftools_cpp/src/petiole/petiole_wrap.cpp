#include "petiole_wrap.h"

#include <opencv2/core.hpp>
#include <stdexcept>

#include "LeafPetiolePointPairFinder.h"

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
