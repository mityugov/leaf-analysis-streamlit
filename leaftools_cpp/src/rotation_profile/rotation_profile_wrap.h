#pragma once

#include "common/numpy_mat.h"

/**
 * @brief Профиль вращения листа.
 *
 * @param leaf_mask Маска (H, W) uint8, лист = ненулевые пиксели.
 * @param center_x, center_y Центр вращения, обычно центр масс листа.
 * @param leaf_area Площадь листа в пикселях (> 0).
 *
 * @return dict: angles — углы 0…180°, jaccard_values — мера Жаккара
 *         для каждого угла, count — число углов (181).
 */
py::dict generate_descriptor(
    const MaskArray& leaf_mask,
    int center_x,
    int center_y,
    int leaf_area);
