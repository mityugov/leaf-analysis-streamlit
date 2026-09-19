#pragma once

#include "./ContourGeodesicDistances.h"
#include "./EuclideanCND.h"
#include "./SVM_Classifier.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>
#include <memory>
#include <optional>
#include <vector>
#include <limits>
#include <string>

// ─────────────────────────────────────────────────────────────────────────────
// Enums
// ─────────────────────────────────────────────────────────────────────────────

enum class ErrorCode
{
    SUCCESS = 0,
    INVALID_INDEXES,
    MIN_GEODESIC_RATIO_INVALID,
    MAX_GEODESIC_RATIO_INVALID,
    NO_WHITE_BETWEEN,
    ZERO_DISTANCE_AB,
    ZERO_DISTANCE_CD,
    MIN_WIDTH_STEM_FAIL,
    STEM_FLATNESS_FAIL,
    INVALID_T_PARAMETERS,
};

/**
 * @brief Результат проверки геодезического отношения G_min / (G_min + G_max).
 */
enum class RatioCheck
{
    TooSmall = -1, ///< Отношение меньше нижнего порога
    OK = 0, ///< Отношение в допустимом диапазоне
    TooLarge = 1, ///< Отношение больше верхнего порога
};

// ─────────────────────────────────────────────────────────────────────────────
// Data structures
// ─────────────────────────────────────────────────────────────────────────────

struct LeafPoints
{
    cv::Point ptA{}, ptB{}, ptS{}, ptC{}, ptD{};
    cv::Point ptSnakeDeA{}, ptSnakeDeB{};
};

struct BestPairIndexes
{
    int best_i_1 = -1;
    int best_j_1 = -1;
    int best_i_2 = -1;
    int best_j_2 = -1;

    bool hasFirstPair()  const { return best_i_1 >= 0 && best_j_1 >= 0; }
    bool hasSecondPair() const { return best_i_2 >= 0 && best_j_2 >= 0; }
};

struct Distances
{
    const double MAX_DE = 70;
    double De = 0.0;
    double D_max = 0.0;
    double D_ab = 0.0;
    double D_cd = 0.0;
};

struct GeodesicParams
{
    GeodesicParams(double total_length) 
        : total_length_(total_length)
    {
    }

    void set(double g_min)
    {
        G_min = g_min;
        G_max = total_length_ - g_min;
        ratio = G_min / (G_min + G_max);
    }

    double getGMin()  const { return G_min; }
    double getGMax()  const { return G_max; }
    double getRatio() const { return ratio; }

private:
    double total_length_ = 0.0;
    double G_min = 0.0;
    double G_max = 0.0;
    double ratio = 0.0;
};

/**
 * @brief Ключевые геодезические расстояния для трёх опорных точек стебля (S, C, D).
 */
struct StemKeyDistances
{
    double G_k = 0.0; ///< Геодезическое расстояние до средней точки S
    double G_c = 0.0; ///< Геодезическое расстояние до точки C (3/4 от B к A)
    double G_d = 0.0; ///< Геодезическое расстояние до точки D (1/4 от B к A)
};

struct Thresholds
{
    // Геодезическое отношение: основной поиск
    double high_threshold = 0.45; ///< Верхняя граница G_min / (G_min + G_max)
    double low_threshold = 0.03; ///< Нижняя граница G_min / (G_min + G_max)

    // Геодезическое отношение: поиск snake De
    double snake_high_threshold = 0.05; ///< Верхняя граница для snake De поиска
    double snake_low_threshold = 0.02; ///< Нижняя граница для snake De поиска

    // Ширина стебля
    int width_threshold = 20; ///< Минимальная суммарная ширина стебля (D_ab + D_cd)

    // Функция phi
    double alpha = 0.3; ///< Вес экспоненциальной составляющей в phi(D)

    // Итоговый скоринг T
    //double w1 = 0.4; ///< Вес T1 в итоговом T
    //double w2 = 0.4; ///< Вес T2 в итоговом T
    //double w3 = 0.2; ///< Вес T3 в итоговом T (умножается на 2)
    double w1 = 0.35; ///< Вес T1 в итоговом T
    double w2 = 0.20; ///< Вес T2 в итоговом T
    double w3 = 0.45; ///< Вес T3 в итоговом T (умножается на 2)


    double T_accept_threshold = 0.65; ///< T >= этого — результат принимается сразу
    double T_reject_threshold = 0.30; ///< T <= этого — результат отклоняется сразу
    double T_epsilon = 0.10; ///< Допуск при сравнении T_res1 и T_res2
    double D_S_rel_tolerance = 0.05; ///< Допуск при поиске второго кандидата: |D_max - D_S1_S2| / D_max

    double component_epsilon = 1e-3; ///< Минимальное ненулевое значение T1/T3
    double T2_min_epsilon = 0.30; ///< Минимальное допустимое значение T2
    double distance_epsilon = 1e-9; ///< Эпсилон для сравнения расстояний с нулём
};

struct T_Parameters
{
    double T1 = 0.0;
    double T2 = 0.0;
    double T3 = 0.0;
    double T  = 0.0;
    double D_ab = 0.0;
    double D_cd = 0.0;

    bool isValid(const Thresholds& thr) const
    {
        return T1 > thr.component_epsilon
            && T2 > thr.T2_min_epsilon
            && T3 > thr.component_epsilon
            && T > 0.0;
    }
};

/**
 * @brief Результат работы compareTAndFindD.
 */
struct FindResult
{
    cv::Point    ptA{};
    cv::Point    ptB{};
    cv::Point    ptS{};
    bool         found = false;
    T_Parameters tp1{};
    T_Parameters tp2{};
    cv::Point    ptA2{};
    cv::Point    ptB2{};
    cv::Point    ptS2{};
};

// ─────────────────────────────────────────────────────────────────────────────
// LeafPetiolePointPairFinder
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Находит пару точек на контуре листа, соответствующих основанию черешка.
 *
 * Алгоритм:
 * 1. Извлекает наибольший контур из бинарного изображения.
 * 2. Строит структуру геодезических расстояний вдоль контура.
 * 3. Перебирает пары точек контура (i, j) и для каждой вычисляет
 *    метрики T1, T2, T3 и итоговый балл T.
 * 4. Выбирает пару с максимальным T, при необходимости использует SVM.
 */
class LeafPetiolePointPairFinder
{
public:
    /**
     * @brief Конструктор.
     * @param binary_image Бинарное изображение листа (CV_8UC1).
     * @param svm_model_path Путь к файлу модели SVM.
     *        Если файл не найден, будет предпринята попытка обучения из CSV.
     * @param svm_csv_path  Путь к CSV-файлу обучающей выборки SVM.
     * @param thresholds    Пользовательские пороговые значения (опционально).
     */
    explicit LeafPetiolePointPairFinder(
        const cv::Mat& binary_image,
        const std::string& svm_model_path = "",
        const std::string& svm_csv_path = "",
        const Thresholds& thresholds = Thresholds{});

    // ── Основные методы ──────────────────────────────────────────────────────

    /**
     * @brief Анализирует пару точек контура по индексам и вычисляет T-параметры.
     * @param i Индекс первой точки.
     * @param j Индекс второй точки.
     * @return Код ошибки или SUCCESS.
     */
    ErrorCode analyzePairPoints(int i, int j);

    /**
     * @brief Выполняет полный перебор пар точек и находит лучшую пару черешка.
     * @return Результат поиска: точки A, B, S и флаг успеха.
     */
    FindResult findBestPair();

    // ── Геттеры ──────────────────────────────────────────────────────────────

    const std::vector<cv::Point>& getContour()   const { return contour_; }
    const Distances& getDistances()  const { return distances_; }
    const LeafPoints& getPoints()     const { return leaf_points_; }
    const T_Parameters& getT()          const { return tparameters_; }
    const BestPairIndexes& getIndexes()    const { return indexes_; }

    // ── Сеттеры порогов ──────────────────────────────────────────────────────

    void setWidthThreshold(int value)
    {
        if (value > 0)
            thresholds_.width_threshold = value;
    }

    void setHighThreshold(double value)
    {
        thresholds_.high_threshold = std::clamp(value, 0.0, 1.0);
    }

    void setLowThreshold(double value)
    {
        thresholds_.low_threshold = std::clamp(value, 0.0, thresholds_.high_threshold);
    }

    void setAlpha(double value)
    {
        thresholds_.alpha = std::clamp(value, 0.0, 1.0);
    }

private:
    // ── Данные ───────────────────────────────────────────────────────────────

    cv::Mat                                  binary_image_;
    std::vector<cv::Point>                   contour_;
    std::unique_ptr<ContourGeodesicDistances> gdc_;
    SVM_Classifier                           svm_;
    double                                   total_length_ = 0.0;

    LeafPoints    leaf_points_;
    Distances     distances_;
    GeodesicParams geodesic_;
    Thresholds    thresholds_;
    T_Parameters  tparameters_;
    BestPairIndexes indexes_;

    // ── Инициализация ─────────────────────────────────────────────────────────

    void findMaxContour();
    void initSVM(const std::string& model_path, const std::string& csv_path);

    // ── Проверки (constraints) ────────────────────────────────────────────────

    bool isValidPoint(const cv::Point& pt) const noexcept;
    bool isValidIndices(int i, int j)       const noexcept;

    /**
     * @brief Проверяет, что отрезок AB проходит через белые пиксели.
     *
     * Использует cv::LineIterator для обхода всех пикселей отрезка,
     * а не только трёх опорных точек.
     */
    bool checkWhiteBetween() const;

    bool checkBorderPixels() const noexcept;

    /**
     * @brief Проверяет геодезическое отношение G_min / (G_min + G_max).
     * @param high_threshold Верхняя граница (по умолчанию — основной порог).
     */
    RatioCheck checkGeodesicRatio(double low_threshold = -1.0, double high_threshold = -1.0) const noexcept;

    bool checkMinWidthStem()  const noexcept;
    bool checkStemFlatness()  const noexcept;

    bool checkTPoint();


    // ── Вычисление метрик ─────────────────────────────────────────────────────

    /**
     * @brief Вычисляет геодезические расстояния до опорных точек стебля.
     *
     * Определяет G_k (середина дуги i→j), G_c и G_d (четверти).
     * Корректно обрабатывает случай j < i (переход через начало контура).
     *
     * @param i Индекс точки A на контуре.
     * @param j Индекс точки B на контуре.
     * @return Структура с G_k, G_c, G_d.
     */
    StemKeyDistances computeStemKeyDistances(size_t i, size_t j) const;

    double phi(double D) const noexcept;

    T_Parameters computeTParameters() const noexcept;

    /**
     * @brief Применяет SVM к набору T-параметров.
     * @return true, если SVM предсказывает положительный класс.
     */
    bool svmPredict(const T_Parameters& tp) const;

    /**
     * @brief Значение decision function SVM (расстояние до гиперплоскости).
     */
    float svmDecision(const T_Parameters& tp);

    // ── Внутренняя структура для хранения кандидата ───────────────────────────

    struct Candidate
    {
        cv::Point    ptA{}, ptB{}, ptS{};
        T_Parameters tp{};
        double       D_ab = 0.0;
        int          idx_i = -1, idx_j = -1;
        bool         valid = false;
    };

    /**
     * @brief Выполняет один проход O(n²) по всем парам точек.
     *
     * @param ref_ptS    Опорная точка S первого кандидата (для поиска второго кандидата).
     * @param D_max      Максимальное D_ab из первого прохода (для фильтрации второго кандидата).
     * @return Лучший найденный кандидат.
     */
    Candidate runScanPass(const cv::Point* ref_ptS = nullptr, double D_max = 0.0);
};