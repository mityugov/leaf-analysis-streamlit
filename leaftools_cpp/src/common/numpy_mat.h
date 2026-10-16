#pragma once

// Преобразования numpy <-> cv::Mat для обёрток *_wrap.cpp.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/core.hpp>

#include <cstring>
#include <stdexcept>

namespace py = pybind11;

/**
 * @brief Тип параметра для бинарной маски из Python.
 *
 * forcecast — массивы других типов (bool, int32 и т.п.) приводятся к uint8.
 * c_style   — строки массива лежат в памяти подряд; срезы вроде mask[::2]
 *             копируются в непрерывный массив.
 */
using MaskArray = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;

/**
 * @brief Копирует маску (H, W) uint8 в cv::Mat типа CV_8UC1.
 *
 * Изменения в возвращённом cv::Mat не влияют на массив в Python.
 *
 * @throws std::runtime_error, если массив не двумерный
 *         (в Python — RuntimeError).
 */
inline cv::Mat maskToMat(const MaskArray& mask)
{
    if (mask.ndim() != 2)
        throw std::runtime_error("mask must be a 2D numpy array (H, W)");

    int rows = static_cast<int>(mask.shape(0));
    int cols = static_cast<int>(mask.shape(1));

    // Строки массива лежат подряд, поэтому данные копируются одним блоком.
    cv::Mat image(rows, cols, CV_8UC1);
    std::memcpy(image.data, mask.data(), static_cast<size_t>(rows) * cols);
    return image;
}

/**
 * @brief Копирует изображение CV_8UC1 или CV_8UC3 в numpy-массив
 *        формы (H, W) или (H, W, 3).
 */
inline py::array matToNumpy(const cv::Mat& image)
{
    if (image.depth() != CV_8U || (image.channels() != 1 && image.channels() != 3))
        throw std::runtime_error("matToNumpy: expected CV_8UC1 or CV_8UC3 image");

    // Часть изображения (ROI) может лежать в памяти с промежутками между строками.
    cv::Mat continuous;
    if (image.isContinuous()) {
        continuous = image;
    } else {
        continuous = image.clone();
    }

    // Конструктор array_t без объекта-владельца копирует данные.
    if (continuous.channels() == 3) {
        return py::array_t<uint8_t>(
            {continuous.rows, continuous.cols, 3},
            {static_cast<size_t>(continuous.step[0]), static_cast<size_t>(continuous.step[1]), sizeof(uint8_t)},
            continuous.ptr<uint8_t>());
    }
    return py::array_t<uint8_t>(
        {continuous.rows, continuous.cols},
        {static_cast<size_t>(continuous.step[0]), sizeof(uint8_t)},
        continuous.ptr<uint8_t>());
}