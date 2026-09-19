#pragma once

#include <opencv2/core.hpp>
#include <vector>

/**
 * @brief Абстрактный базовый класс для вычисления расстояний между соседними точками контура.
 *
 * Этот класс предоставляет интерфейс для вычисления расстояний между соседними точками
 * вдоль контура. Контур может рассматриваться как замкнутый (кольцо) или незамкнутый
 * в зависимости от параметра `closed`.
 */
class IContourNeighborDistances
{
protected:
    std::vector<cv::Point> contour_; ///< Входной контур в виде вектора точек.
    bool closed_;                    ///< Флаг, указывающий, является ли контур замкнутым (соединяется последняя точка с первой).

public:
    /**
     * @brief Конструктор IContourNeighborDistances.
     * @param contour Контур, представленный вектором cv::Point.
     * @param closed Если true, контур рассматривается как замкнутый (последняя точка соединяется с первой).
     */
    IContourNeighborDistances(const std::vector<cv::Point>& contour, bool closed = true) : contour_(contour), closed_(closed) {}

    /**
     * @brief Виртуальный деструктор.
     */
    virtual ~IContourNeighborDistances() = default;

    /**
     * @brief Вычисляет и возвращает расстояния между соседними точками контура.
     * @return Вектор расстояний (double) между последовательными точками.
     */
    virtual std::vector<double> calculate() const = 0;

    /**
     * @brief Возвращает флаг замкнутости контура.
     * @return true, если контур замкнутый.
     */
    bool isClosed() const { return closed_; }
};
