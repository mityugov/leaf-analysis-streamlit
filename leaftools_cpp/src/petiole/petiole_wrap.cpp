#include "petiole_wrap.h"

#include "LeafPetiolePointPairFinder.h"

py::dict find_petiole_points(
        const MaskArray& leaf_mask,
        const std::string& svm_model_path,
        const std::string& svm_csv_path) {
    cv::Mat leaf_image = maskToMat(leaf_mask);

    LeafPetiolePointPairFinder finder(leaf_image, svm_model_path, svm_csv_path);
    FindResult best_pair = finder.findBestPair();

    // При found = false координаты содержат нули или точки отклонённого кандидата.
    py::dict result;
    result["found"] = best_pair.found;
    result["ax"] = best_pair.ptA.x;  result["ay"] = best_pair.ptA.y;
    result["bx"] = best_pair.ptB.x;  result["by"] = best_pair.ptB.y;
    result["sx"] = best_pair.ptS.x;  result["sy"] = best_pair.ptS.y;
    return result;
}
