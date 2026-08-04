#pragma once

#include "./IContourNeighborDistances.h"
#include <algorithm>
#include <memory>
#include <vector>

/**
 * @brief Класс для вычисления геодезических расстояний вдоль контура.
 *
 * Вычисляет расстояния по кратчайшему пути вдоль контура между любыми двумя точками
 * на основе префиксных сумм расстояний между соседними точками.
 *
 * Внутреннее представление:
 *   prefixSums[0] = 0
 *   prefixSums[k] = сумма расстояний от точки 0 до точки k
 *
 * Размер prefixSums:
 *   - незамкнутый контур из n точек: n элементов (индексы 0..n-1)
 *   - замкнутый контур из n точек:   n+1 элементов (последний — полный периметр)
 *
 * Допустимые индексы точек: 0 .. getPointCount()-1
 */
class ContourGeodesicDistances
{
private:
    std::vector<double> prefixSums_; ///< Префиксные суммы расстояний между соседними точками.
    bool closed_;                    ///< Флаг замкнутости контура.

    /**
     * @brief Инициализирует префиксные суммы на основе расстояний между соседями.
     *
     * Для незамкнутого контура из n точек: n расстояний → n+1 префиксных сумм,
     * но последний элемент не несёт смысловой нагрузки как точка контура.
     * Поэтому мы храним ровно n элементов (prefixSums[0]=0, ..., prefixSums[n-1]).
     *
     * Для замкнутого контура из n точек: n расстояний (включая last→first) →
     * n+1 префиксных сумм, где prefixSums[n] = полный периметр.
     *
     * @param neighborDistances Вектор расстояний между соседними точками.
     *        Размер: (n-1) для незамкнутого или n для замкнутого контура.
     * @param closed Флаг замкнутости.
     */
    void initPrefixSums(const std::vector<double>& neighborDistances, bool closed)
    {
        closed_ = closed;
        prefixSums_.clear();

        if (neighborDistances.empty())
            return;

        // Количество точек контура:
        // - незамкнутый: neighborDistances.size() = n-1  → n = size+1 точек
        // - замкнутый:   neighborDistances.size() = n    → n = size точек
        // Нам нужно n+1 префиксных сумм для замкнутого (включая периметр)
        // и n префиксных сумм для незамкнутого (от 0 до n-1 включительно).
        const size_t reserveSize = neighborDistances.size() + 1;
        prefixSums_.reserve(reserveSize);
        prefixSums_.push_back(0.0);

        for (double d : neighborDistances)
            prefixSums_.push_back(prefixSums_.back() + d);

        // Для незамкнутого контура последний элемент prefixSums_ соответствует
        // "расстоянию за последней точкой" — он не нужен, удаляем его.
        // Пример: 3 точки, 2 расстояния → prefixSums_ = {0, d01, d01+d12}
        //         Индексы точек 0,1,2 → нужны prefixSums_[0..2], их уже 3.
        // Для замкнутого: 3 точки, 3 расстояния → prefixSums_ = {0, d01, d01+d12, периметр}
        //         Индексы точек 0,1,2 → нужны prefixSums_[0..2], плюс [3] = периметр.
        // Таким образом, для незамкнутого размер prefixSums_ уже правильный (n),
        // ничего удалять не нужно — всё корректно.
    }

    /**
     * @brief Возвращает количество точек контура.
     *
     * Для замкнутого:   prefixSums_.size() - 1  (последний элемент — периметр)
     * Для незамкнутого: prefixSums_.size()       (каждый элемент — точка)
     */
    size_t pointCount() const noexcept
    {
        if (prefixSums_.empty())
            return 0;
        return closed_ ? prefixSums_.size() - 1 : prefixSums_.size();
    }

    /**
     * @brief Проверяет, является ли индекс допустимым индексом точки контура.
     */
    bool isValidIndex(size_t i) const noexcept
    {
        return i < pointCount();
    }

public:
    /**
     * @brief Конструктор из вектора расстояний между соседними точками.
     *
     * @param neighborDistances Вектор расстояний:
     *        - незамкнутый контур из n точек: n-1 расстояний
     *        - замкнутый контур из n точек:   n расстояний (последнее — last→first)
     * @param closed Если true, контур замкнутый.
     */
    ContourGeodesicDistances(const std::vector<double>& neighborDistances, bool closed = true)
    {
        initPrefixSums(neighborDistances, closed);
    }

    /**
     * @brief Конструктор из калькулятора расстояний между соседними точками.
     *
     * @param ndc Умный указатель на IContourNeighborDistances.
     */
    explicit ContourGeodesicDistances(std::unique_ptr<IContourNeighborDistances> ndc)
        : ContourGeodesicDistances(ndc->calculate(), ndc->isClosed())
    {
    }

    /**
     * @brief Возвращает геодезическое расстояние от начала контура до точки i.
     *
     * @param i Индекс точки (0 .. getPointCount()-1).
     * @return Расстояние от точки 0 до точки i, или -1.0 при неверном индексе.
     */
    double getGeodesicDistance(size_t i) const noexcept
    {
        if (!isValidIndex(i))
            return -1.0;
        return prefixSums_[i]; // prefixSums_[0] всегда 0, поэтому вычитание не нужно
    }

    /**
     * @brief Вычисляет геодезическое расстояние от точки i до точки j
     *        в направлении i → j (по возрастанию индексов).
     *
     * - j >= i: прямой путь i → j
     * - j < i, замкнутый: циклический путь i → конец контура → начало → j
     * - j < i, незамкнутый: возвращает -1.0 (обратный путь недопустим)
     *
     * @param i Индекс начальной точки.
     * @param j Индекс конечной точки.
     * @return Геодезическое расстояние или -1.0 при недопустимых параметрах.
     */
    double getGeodesicDistance(size_t i, size_t j) const noexcept
    {
        if (!isValidIndex(i) || !isValidIndex(j))
            return -1.0;

        if (i == j)
            return 0.0;

        if (j > i)
        {
            // Прямой путь i → j
            return prefixSums_[j] - prefixSums_[i];
        }

        // j < i
        if (!closed_)
            return -1.0; // Обратный путь недопустим для незамкнутого контура

        // Циклический путь: i → конец → начало → j
        // prefixSums_.back() = полный периметр
        return (prefixSums_.back() - prefixSums_[i]) + prefixSums_[j];
    }

    /**
     * @brief Находит индекс точки контура, ближайшей к заданному расстоянию от начала.
     *
     * Использует бинарный поиск по отсортированным префиксным суммам.
     * Для замкнутого контура при dist >= периметра возвращает индекс 0.
     * Для незамкнутого при dist >= длины контура возвращает последний индекс.
     *
     * @param dist Расстояние от начала контура (>= 0).
     * @return Индекс ближайшей точки.
     */
    //size_t getIndexByDistance(double dist) const noexcept
    //{
    //    return getIndexByDistanceInRange(dist, 0, pointCount());
    //}
    size_t getIndexByDistance(double dist)
    {
        if (prefixSums_.empty() || dist < 0.0)
            return 0;

        if (dist >= prefixSums_.back())
            return (closed_) ? 0 : prefixSums_.size() - 1;

        /// Бинарный поиск для отсортированного массива
        auto it = std::lower_bound(prefixSums_.begin(), prefixSums_.end(), dist);

        if (it == prefixSums_.begin())
            return 0;

        if (it == prefixSums_.end())
            return (closed_) ? 0 : prefixSums_.size() - 1;

        /// Сравниваем расстояния до двух соседних элементов
        size_t idx_right = std::distance(prefixSums_.begin(), it) - 1;

        if (idx_right == 0) {
            return 0;
        }

        size_t idx_left = idx_right - 1;

        double dist_left = std::abs(prefixSums_[idx_left] - dist);
        double dist_right = std::abs(prefixSums_[idx_right] - dist);

        return (dist_left <= dist_right) ? idx_left : idx_right;
    }

    /**
     * @brief Находит индекс точки в диапазоне [left, right), ближайшей к заданному расстоянию.
     *
     * Позволяет ограничить поиск подмножеством точек контура.
     * Диапазон задаётся в терминах индексов точек (не prefixSums_).
     *
     * @param dist   Расстояние от начала контура (>= 0).
     * @param left   Левая граница диапазона индексов точек (включительно).
     * @param right  Правая граница диапазона индексов точек (не включительно).
     *               Если right > getPointCount(), обрезается до getPointCount().
     * @return Индекс ближайшей точки в диапазоне [left, right).
     *         Возвращает left, если диапазон пуст или dist <= prefixSums_[left].
     */
    size_t getIndexByDistanceInRange(double dist, size_t left, size_t right) const noexcept
    {
        const size_t n = pointCount();

        if (n == 0)
            return 0;

        // Нормализуем границы до допустимого диапазона точек [0, n)
        left = std::min(left, n - 1);
        right = std::min(right, n);

        if (left >= right)
            return left;

        if (dist <= prefixSums_[left])
            return left;

        // Граница справа: для последней точки prefixSums_[right-1]
        if (dist >= prefixSums_[right - 1])
            return closed_ ? left : right - 1;

        // Бинарный поиск в подмассиве prefixSums_[left..right-1]
        // lower_bound вернёт итератор на первый элемент >= dist
        auto begin = prefixSums_.cbegin() + static_cast<std::ptrdiff_t>(left);
        auto end = prefixSums_.cbegin() + static_cast<std::ptrdiff_t>(right);
        auto it = std::lower_bound(begin, end, dist);

        // it указывает на первый элемент >= dist (правый кандидат)
        // std::prev(it) — последний элемент < dist (левый кандидат)
        // it != begin гарантировано, так как dist > prefixSums_[left]
        const size_t idxRight = left + static_cast<size_t>(std::distance(begin, it));
        const size_t idxLeft = idxRight - 1;

        const double distToLeft = dist - prefixSums_[idxLeft];
        const double distToRight = prefixSums_[idxRight] - dist;

        return (distToLeft <= distToRight) ? idxLeft : idxRight;
    }

    /**
     * @brief Возвращает флаг замкнутости контура.
     */
    bool isClosed() const noexcept { return closed_; }

    /**
     * @brief Возвращает полную длину контура (периметр для замкнутого).
     */
    double getTotalLength() const noexcept
    {
        return prefixSums_.empty() ? 0.0 : prefixSums_.back();
    }

    /**
     * @brief Возвращает количество точек в контуре.
     */
    size_t getPointCount() const noexcept { return pointCount(); }
};