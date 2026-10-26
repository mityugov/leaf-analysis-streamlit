#pragma once

#include "common/numpy_mat.h"

/**
 * @brief Поиск криволинейной оси симметрии листа.
 *
 * @param leaf_mask        Маска (H, W) uint8, лист = 255 на чёрном фоне.
 * @param smoothing_lambda Параметр сглаживания оси, 0…1 (в Python — lambda_).
 * @param start_x, start_y Начальная точка оси; -1 — поиск автоматически.
 *
 * @return dict: found, jaccard_straightened, jaccard_original; при found = true
 *         также axis, straightened, reflected — изображения BGR в координатах
 *         маски, обрезанной по рамке листа; axis_points — точки оси (N, 2)
 *         float32 (x, y) в координатах входной маски.
 */
py::dict find_symmetry(
    const MaskArray& leaf_mask,
    double smoothing_lambda,
    int start_x,
    int start_y);
