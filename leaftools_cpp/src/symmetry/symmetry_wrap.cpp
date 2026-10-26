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

    // Точки оси в координатах входной маски: алгоритм работает с маской,
    // обрезанной по рамке листа, поэтому прибавляем сдвиг этой рамки.
    const size_t point_count = best_axis.curvilinearAxisAtoms.size();
    py::array_t<float> axis_points({point_count, static_cast<size_t>(2)});
    auto points = axis_points.mutable_unchecked<2>();
    for (size_t i = 0; i < point_count; i++) {
        const cv::Point2f& atom = best_axis.curvilinearAxisAtoms[i];
        points(i, 0) = atom.x + search_input.offset_.x;
        points(i, 1) = atom.y + search_input.offset_.y;
    }
    result["axis_points"] = axis_points;
    result["straightened"] = matToNumpy(best_axis.straightened_);
    result["reflected"] = matToNumpy(best_axis.reflected_);
    return result;
}