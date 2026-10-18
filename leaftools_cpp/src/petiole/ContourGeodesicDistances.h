#pragma once

#include "./IContourNeighborDistances.h"
#include <algorithm>
#include <memory>
#include <vector>

/**
 * @brief Геодезические расстояния вдоль контура (длина пути по контуру).
 *
 * Расстояние от точки i до точки j считается в направлении обхода контура
 * (по возрастанию индексов). Для замкнутого контура путь при j < i проходит
 * через конец контура и его начало.
 *
 * Точки нумеруются от 0 до n-1, где n — количество точек контура.
 * Расстояния вычисляются через префиксные суммы (см. initPrefixSums).
 */
class ContourGeodesicDistances
{
private:
    std::vector<double> prefixSums_; ///< Префиксные суммы расстояний между соседними точками.
    bool closed_;                    ///< Флаг замкнутости контура.

    /**
     * @brief Заполняет prefixSums_ по расстояниям между соседними точками.
     *
     * prefixSums_[k] — длина пути по контуру от точки 0 до точки k:
     *   prefixSums_[0] = 0
     *   prefixSums_[k] = prefixSums_[k-1] + расстояние от точки k-1 до точки k
     *
     * Точки нумеруются от 0 до n-1, индекс точки k совпадает с индексом
     * элемента prefixSums_[k].
     *
     * Размер prefixSums_ для контура из n точек:
     *   - незамкнутый: n элементов;
     *   - замкнутый:   n+1 элементов (последний добавляет отрезок
     *                  от последней точки до первой).
     *
     * Пример: замкнутый контур из 3 точек с расстояниями d01, d12, d20:
     *   prefixSums_ = {0, d01, d01+d12, d01+d12+d20}
     *
     * @param neighborDistances Расстояния между соседними точками.
     * @param closed true, если контур замкнутый.
     */
    void initPrefixSums(const std::vector<double>& neighborDistances, bool closed)
    {
        closed_ = closed;
        prefixSums_.clear();

        if (neighborDistances.empty())
            return;

        // Элементов на один больше, чем расстояний: первый элемент равен 0.
        const size_t reserveSize = neighborDistances.size() + 1;
        prefixSums_.reserve(reserveSize);
        prefixSums_.push_back(0.0);

        for (double d : neighborDistances)
            prefixSums_.push_back(prefixSums_.back() + d);
    }

    /**
     * @brief Возвращает количество точек контура.
     *
     * У замкнутого контура последний элемент prefixSums_ добавляет отрезок
     * от последней точки до первой и не соответствует новой точке,
     * поэтому он не учитывается.
     *
     * Для замкнутого:   prefixSums_.size() - 1
     * Для незамкнутого: prefixSums_.size()
     */
    size_t getPointCount() const noexcept
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
        return i < getPointCount();
    }

public:
    /**
     * @brief Конструктор из вектора расстояний между соседними точками.
     *
     * @param neighborDistances Вектор расстояний:
     *        - незамкнутый контур из n точек: n-1 расстояний
     *        - замкнутый контур из n точек:   n расстояний (последнее — от последней точки до первой)
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
     * @param i Индекс точки, от 0 до n-1.
     * @return Расстояние от точки 0 до точки i или -1.0 при неверном индексе.
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
        // prefixSums_.back() — длина полного обхода контура
        return (prefixSums_.back() - prefixSums_[i]) + prefixSums_[j];
    }

    /**
     * @brief Находит индекс точки контура, ближайшей к заданному расстоянию от начала.
     *
     * Использует бинарный поиск по отсортированным префиксным суммам.
     * Для замкнутого контура при dist >= длины контура возвращает индекс 0.
     * Для незамкнутого при dist >= длины контура возвращает последний индекс.
     *
     * @param dist Расстояние от начала контура (>= 0).
     * @return Индекс ближайшей точки.
     */
    //size_t getIndexByDistance(double dist) const noexcept
    //{
    //    return getIndexByDistanceInRange(dist, 0, getPointCount());
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
        const size_t n = getPointCount();

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
     * @brief Возвращает полную длину контура (для замкнутого — с отрезком от последней точки до первой).
     */
    double getTotalLength() const noexcept
    {
        return prefixSums_.empty() ? 0.0 : prefixSums_.back();
    }
};