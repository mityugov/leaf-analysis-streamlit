#pragma once

#include <opencv2/core.hpp>
#include <vector>

/**
 * @brief јбстрактный базовый класс дл€ вычислени€ рассто€ний между соседними точками контура.
 * 
 * Ётот класс предоставл€ет интерфейс дл€ вычислени€ рассто€ний между соседними точками
 * вдоль контура.  онтур может рассматриватьс€ как замкнутый (кольцо) или незамкнутый
 * в зависимости от параметра `closed`.
 */
class IContourNeighborDistances
{
protected:
    std::vector<cv::Point> contour_; ///< ¬ходной контур в виде вектора точек.
    bool closed_;                    ///< ‘лаг, указывающий, €вл€етс€ ли контур замкнутым (соедин€етс€ последн€€ точка с первой).

public:
    /**
     * @brief  онструктор IContourNeighborDistances.
     * @param contour  онтур, представленный вектором cv::Point.
     * @param closed ≈сли true, контур рассматриваетс€ как замкнутый (последн€€ точка соедин€етс€ с первой).
     */
    IContourNeighborDistances(const std::vector<cv::Point>& contour, bool closed = true) : contour_(contour), closed_(closed) {}

    /**
     * @brief ¬иртуальный деструктор.
     */
    virtual ~IContourNeighborDistances() = default;

    /**
     * @brief ¬ычисл€ет и возвращает рассто€ни€ между соседними точками контура.
     * @return ¬ектор рассто€ний (double) между последовательными точками.
     */
    virtual std::vector<double> calculate() const = 0;

    /**
     * @brief ¬озвращает флаг замкнутости контура.
     * @return true, если контур замкнутый.
     */
    bool isClosed() const { return closed_; }
};
