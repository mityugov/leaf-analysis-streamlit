#pragma once

#include "./IContourNeighborDistances.h"

/**
 * @brief Класс для вычисления евклидовых расстояний между соседними точками контура.
 *
 * Реализует интерфейс IContourNeighborDistances, вычисляя расстояния между последовательными
 * точками контура с использованием евклидова расстояния. Поддерживает как замкнутые, так и
 * незамкнутые контуры.
 */
class EuclideanCND : public IContourNeighborDistances
{
public:
    /**
     * @brief Конструктор EuclideanCND.
     * @param contour Контур в виде вектора точек cv::Point.
     * @param closed Если true, контур считается замкнутым (добавляется расстояние от последней точки к первой).
     */
    EuclideanCND(const std::vector<cv::Point>& contour, bool closed = true)
        : IContourNeighborDistances(contour, closed)
    {
    }

    /**
     * @brief Вычисляет евклидовы расстояния между соседними точками контура.
     *
     * Для каждой пары соседних точек вычисляется евклидово расстояние с помощью cv::norm().
     * При замкнутом контуре добавляется дополнительное расстояние от последней точки к первой.
     *
     * @return Вектор расстояний между соседними точками. Пустой вектор, если в контуре менее 2 точек.
     */
    std::vector<double> calculate() const override
    {
        size_t n = contour_.size();

        std::vector<double> distances{};

        /// Проверяем, достаточно ли точек для вычисления расстояний
        if (n < 2)
            return distances;

        /// Резервируем память для оптимизации
        if (closed_)
            distances.reserve(n);
        else
            distances.reserve(n - 1);

        /// Вычисляем расстояния между последовательными точками
        for (size_t i = 0; i + 1 < n; i++)
        {
            distances.push_back(cv::norm(contour_[i] - contour_[i + 1]));
        }

        /// Для замкнутого контура добавляем расстояние от последней к первой точке
        if (closed_)
            distances.push_back(cv::norm(contour_.back() - contour_.front()));

        return distances;
    }
};
