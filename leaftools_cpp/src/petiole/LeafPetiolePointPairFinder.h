#pragma once

#include "./ContourGeodesicDistances.h"
#include "./SVM_Classifier.h"

#include <opencv2/core.hpp>
#include <memory>
#include <string>
#include <vector>

/**
 * @brief Результат проверки геодезического отношения G_min / (G_min + G_max).
 */
enum class RatioCheck
{
    TooSmall = -1, ///< Отношение меньше нижнего порога
    OK = 0, ///< Отношение в допустимом диапазоне
    TooLarge = 1, ///< Отношение больше верхнего порога
};

struct LeafPoints
{
    cv::Point ptA{}, ptB{}, ptS{}, ptC{}, ptD{};
};

struct Distances
{
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

    double max_De = 70.0; ///< De принимается, только если оно меньше этого значения

    // Ширина стебля
    int width_threshold = 20; ///< Минимальная суммарная ширина стебля (D_ab + D_cd)

    // Проверки точек
    double border_offset = 2.0;   ///< Отступ от A и B вдоль AB в checkBorderPixels, пикселей
    double t_point_offset = 0.05; ///< Сдвиг точки T за S в checkTPoint, доля длины AS

    // Функция phi
    double alpha = 0.3; ///< Вес экспоненциальной составляющей в phi(D)

    // Итоговый скоринг T
    double w1 = 0.35; ///< Вес T1 в итоговом T
    double w2 = 0.20; ///< Вес T2 в итоговом T
    double w3 = 0.45; ///< Вес T3 в итоговом T (умножается на 2)


    double T_accept_threshold = 0.65; ///< T >= этого — результат принимается сразу
    double T_reject_threshold = 0.30; ///< T <= этого — результат отклоняется сразу
    double T_epsilon = 0.10; ///< Допуск при сравнении T_res1 и T_res2
    double D_S_rel_tolerance = 0.05; ///< Допуск при поиске второго кандидата: |D_max - D_S1_S2| / D_max

    double component_epsilon = 1e-3; ///< Минимальное ненулевое значение T1/T3
    double T2_min = 0.30; ///< Минимальное допустимое значение T2
    double distance_epsilon = 1e-9; ///< Эпсилон для сравнения расстояний с нулём
};

struct T_Parameters
{
    double T1 = 0.0;
    double T2 = 0.0;
    double T3 = 0.0;
    double T  = 0.0;

    bool isValid(const Thresholds& thr) const
    {
        return T1 > thr.component_epsilon
            && T2 > thr.T2_min
            && T3 > thr.component_epsilon
            && T > 0.0;
    }
};

/**
 * @brief Результат findBestPair.
 *
 * A и B — края черешка на контуре, S — середина дуги контура между ними
 * (кончик черешка). При found = false точки содержат нули или
 * отклонённого кандидата.
 */
struct FindResult
{
    cv::Point    ptA{};
    cv::Point    ptB{};
    cv::Point    ptS{};
    bool         found = false;
};

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

    /**
     * @brief Выполняет полный перебор пар точек и находит лучшую пару черешка.
     * @return Результат поиска: точки A, B, S и флаг успеха.
     */
    FindResult findBestPair();

private:
    cv::Mat                                  binary_image_;
    std::vector<cv::Point>                   contour_;
    std::unique_ptr<ContourGeodesicDistances> gdc_;
    SVM_Classifier                           svm_;
    double                                   total_length_ = 0.0;

    LeafPoints    leaf_points_;
    Distances     distances_;
    GeodesicParams geodesic_;
    Thresholds    thresholds_;

    void findMaxContour();

    /**
     * @brief Загружает модель SVM или обучает её из CSV.
     *
     * Сначала загружается модель из model_path. Если загрузить не удалось,
     * модель обучается по csv_path и сохраняется в model_path.
     * Если оба пути пустые или неверные, SVM остаётся необученным:
     * svmPredict() возвращает false, и кандидат в «серой зоне» отклоняется.
     */
    void initSVM(const std::string& model_path, const std::string& csv_path);

    bool isValidPoint(const cv::Point& pt) const noexcept;
    bool isValidIndices(int i, int j)       const noexcept;

    /**
     * @brief Проверяет, что отрезок AB проходит внутри листа.
     *
     * Проверяются три точки отрезка: середина и середины его половин.
     * Все три должны быть белыми.
     */
    bool checkWhiteBetween() const;

    /**
     * @brief Проверяет, что точки рядом с A и B на отрезке AB лежат внутри листа.
     *
     * От A к B и от B к A делается шаг на 2 пикселя; оба пикселя должны быть белыми.
     */
    bool checkBorderPixels() const noexcept;

    /**
     * @brief Сравнивает геодезическое отношение G_min / (G_min + G_max) с границами.
     */
    RatioCheck checkGeodesicRatio(double low_threshold, double high_threshold) const noexcept;

    bool checkMinWidthStem()  const noexcept;
    bool checkStemFlatness()  const noexcept;

    bool checkTPoint() const;


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

    /**
     * @brief Находит точки S, C, D для пары (i, j) и проверяет форму стебля.
     *
     * Перед вызовом должны быть заданы leaf_points_.ptA, ptB,
     * distances_.D_ab и geodesic_ для этой пары.
     * Заполняет leaf_points_.ptS, ptC, ptD и distances_.D_cd.
     *
     * @return true, если пара прошла все проверки: плоскость стебля,
     *         ширина, граничные пиксели и точка T.
     */
    bool checkStemPoints(int i, int j);

    double phi(double D) const noexcept;

    T_Parameters computeTParameters() const noexcept;

    /**
     * @brief Применяет SVM к признакам кандидата (T1, T2, T3).
     *
     * Используется в findBestPair, когда итоговый балл T лежит между
     * T_reject_threshold и T_accept_threshold.
     *
     * @return true, если SVM относит кандидата к классу +1 (черешок).
     *         false — класс -1 или SVM не обучен.
     */
    bool svmPredict(const T_Parameters& tp) const;

    /**
     * @brief Значение decision function SVM (расстояние до гиперплоскости).
     *
     * Используется только по модулю: чем больше |значение|, тем дальше
     * кандидат от границы классов.
     */
    float svmDecision(const T_Parameters& tp) const;

    struct Candidate
    {
        cv::Point    ptA{}, ptB{}, ptS{};
        T_Parameters tp{};
        bool         valid = false;
    };

    /// Число точек контура, при котором перебор ещё идёт по каждой точке.
    /// Для более длинных контуров берётся каждая k-я точка.
    static constexpr int max_scan_points = 10000;

    FindResult makeResult(const Candidate& candidate, bool found) const;

    /**
     * @brief Выполняет один проход O(n²) по всем парам точек.
     *
     * @param ref_ptS    Опорная точка S первого кандидата (для поиска второго кандидата).
     * @param D_max      Максимальное D_ab из первого прохода (для фильтрации второго кандидата).
     * @return Лучший найденный кандидат.
     */
    Candidate runScanPass(const cv::Point* ref_ptS = nullptr, double D_max = 0.0);
};