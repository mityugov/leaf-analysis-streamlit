#include "symmetry_wrap.h"

#include <stdexcept>

#include "CurvilinearSymmetryBase.h"

py::dict find_symmetry(
        const MaskArray& leaf_mask,
        double smoothing_lambda,
        int start_x,
        int start_y) {

    if (smoothing_lambda < 0.0 || smoothing_lambda > 1.0)
        throw std::runtime_error("lambda must be in [0, 1]");

    // Входные параметры алгоритма.
    Data search_input;
    search_input.image_ = maskToMat(leaf_mask);
    search_input.area_ = cv::countNonZero(search_input.image_);
    search_input.lambda_ = smoothing_lambda;
    search_input.stepLength = 0;
    search_input.out_path_ = "";
    // При пустом in_path_ Data::saveResultImages не записывает файлы.
    search_input.in_path_ = "";

    if (start_x >= 0 && start_y >= 0) {
        search_input.start_point_ = cv::Point2f(static_cast<float>(start_x),
                                                static_cast<float>(start_y));
        search_input.definedInitialPoint = true;
    } else {
        search_input.start_point_ = cv::Point2f(-1, -1);
        search_input.definedInitialPoint = false;
    }

    // Обрезает маску по рамке листа и находит внешний контур.
    if (search_input.takeInfoFromImage() != 0)
        throw std::runtime_error("contour not found in mask");

    GreedySearch search;
    Curvilinear best_axis;
    search.processing(search_input, best_axis);

    // Картинка оси заполняется только при успешном поиске (CurvilinearSymmetryBase::Draw).
    bool found = !best_axis.curvilinearAxisAtoms.empty()
                 && !best_axis.curvilinear_axis_.empty();

    py::dict result;
    result["found"] = found;

    if (!found) {
        result["jaccard_straightened"] = 0.0;
        result["jaccard_original"] = 0.0;
        return result;
    }

    result["jaccard_straightened"] = best_axis.jaccard_index_with_straightened_shape_area_;
    result["jaccard_original"] = best_axis.jaccard_index_with_original_shape_area_;
    result["axis"] = matToNumpy(best_axis.curvilinear_axis_);
    result["straightened"] = matToNumpy(best_axis.straightened_);
    result["reflected"] = matToNumpy(best_axis.reflected_);
    return result;
}
