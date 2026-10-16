#include "rotation_profile_wrap.h"

#include <map>
#include <stdexcept>

#include "RotationProfile.h"

py::dict generate_descriptor(
        const MaskArray& leaf_mask,
        int center_x,
        int center_y,
        int leaf_area) {
    cv::Mat leaf_image = maskToMat(leaf_mask);

    if (leaf_area <= 0) {
        throw std::runtime_error("area must be > 0");
    }

    // Ключ — угол в градусах, значение — мера Жаккара.
    std::map<double, double> jaccard_by_angle =
        generateDescriptorSequence(leaf_image, cv::Point(center_x, center_y), leaf_area);

    // Углы и значения меры записываются в два отдельных numpy-массива.
    const size_t angle_count = jaccard_by_angle.size();
    auto angles = py::array_t<double>(angle_count);
    auto jaccard_values = py::array_t<double>(angle_count);
    double* angles_data = static_cast<double*>(angles.request().ptr);
    double* jaccard_data = static_cast<double*>(jaccard_values.request().ptr);

    size_t i = 0;
    for (const auto& item : jaccard_by_angle) {
        angles_data[i] = item.first;
        jaccard_data[i] = item.second;
        ++i;
    }

    py::dict result;
    result["angles"] = angles;
    result["jaccard_values"] = jaccard_values;
    result["count"] = static_cast<int>(angle_count);
    return result;
}
