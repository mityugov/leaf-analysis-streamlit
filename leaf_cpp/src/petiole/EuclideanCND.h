#pragma once

#include "./IContourNeighborDistances.h"

/**
 * @brief  ласс дл€ вычислени€ евклидовых рассто€ний между соседними точками контура.
 * 
 * –еализует интерфейс IContourNeighborDistances, вычисл€€ рассто€ни€ между последовательными
 * точками контура с использованием евклидова рассто€ни€. ѕоддерживает как замкнутые, так и
 * незамкнутые контуры.
 */
class EuclideanCND : public IContourNeighborDistances
{
public:
    /**
     * @brief  онструктор EuclideanCND.
     * @param contour  онтур в виде вектора точек cv::Point.
     * @param closed ≈сли true, контур считаетс€ замкнутым (добавл€етс€ рассто€ние от последней точки к первой).
     */
    EuclideanCND(const std::vector<cv::Point>& contour, bool closed = true)
        : IContourNeighborDistances(contour, closed)
    {
    }

    /**
     * @brief ¬ычисл€ет евклидовы рассто€ни€ между соседними точками контура.
     *
     * ƒл€ каждой пары соседних точек вычисл€етс€ евклидово рассто€ние с помощью cv::norm().
     * ѕри замкнутом контуре добавл€етс€ дополнительное рассто€ние от последней точки к первой.
     *
     * @return ¬ектор рассто€ний между соседними точками. ѕустой вектор, если в контуре менее 2 точек.
     */
    std::vector<double> calculate() const override
    {
        size_t n = contour_.size();

        std::vector<double> distances{};

        /// ѕровер€ем, достаточно ли точек дл€ вычислени€ рассто€ний
        if (n < 2)
            return distances;

        /// –езервируем пам€ть дл€ оптимизации
        if (closed_)
            distances.reserve(n);
        else
            distances.reserve(n - 1);

        /// ¬ычисл€ем рассто€ни€ между последовательными точками
        for (size_t i = 0; i + 1 < n; i++)
        {
            distances.push_back(cv::norm(contour_[i] - contour_[i + 1]));
        }

        /// ƒл€ замкнутого контура добавл€ем рассто€ние от последней к первой точке
        if (closed_)
            distances.push_back(cv::norm(contour_.back() - contour_.front()));

        return distances;
    }
};
