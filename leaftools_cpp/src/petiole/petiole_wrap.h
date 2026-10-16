#pragma once

#include "common/numpy_mat.h"

#include <string>

/**
 * @brief Поиск точек основания черешка.
 *
 * @param leaf_mask      Маска (H, W) uint8, лист = 255 на чёрном фоне.
 * @param svm_model_path Путь к модели SVM (.xml); пустая строка — без загрузки.
 * @param svm_csv_path   Путь к CSV для обучения SVM, если модель не загружена.
 *
 * @return dict: found — найден ли черешок; ax, ay и bx, by — края черешка
 *         на контуре; sx, sy — середина дуги контура между ними.
 */
py::dict find_petiole_points(
    const MaskArray& leaf_mask,
    const std::string& svm_model_path,
    const std::string& svm_csv_path);
