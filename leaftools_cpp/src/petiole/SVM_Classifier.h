#pragma once

#include <opencv2/ml.hpp>
#include <string>
#include <vector>

/**
 * @brief Обёртка над cv::ml::SVM для бинарной классификации пар точек черешка.
 *
 * Ожидаемый формат меток: +1 (положительный класс) и -1 (отрицательный).
 * Признаки: (T1, T2, T3) — вещественные числа в диапазоне [0, 1].
 *
 * Порядок использования:
 *   1. Создать объект с путём к модели и (опционально) к CSV.
 *   2. Вызвать loadModel() или loadCSV() + train().
 *   3. Использовать predict() и decisionFunction().
 */
class SVM_Classifier
{
public:
    /**
     * @brief Конструктор.
     * @param model_path Путь к файлу модели (.xml/.yaml). По этому пути
     *        loadModel() загружает модель, а train() сохраняет обученную.
     *        Пустая строка — модель не загружается и не сохраняется.
     */
    explicit SVM_Classifier(const std::string& model_path = "");

    /**
     * @brief Загружает обучающую выборку из CSV-файла.
     *
     * Ожидаемый формат (разделитель ';', первая строка — заголовок):
     *   Class;T1;T2;T3
     *   1;0.8;0.6;0.4
     *  -1;0.1;0.2;0.3
     *
     * @param csv_path Путь к CSV-файлу.
     * @return true при успешной загрузке.
     */
    bool loadCSV(const std::string& csv_path);

    /**
     * @brief Загружает обученную модель SVM из файла.
     * @param model_path Путь к файлу модели. Если пуст — используется model_path_ из конструктора.
     * @return true при успешной загрузке.
     */
    bool loadModel(const std::string& model_path = "");

    /**
     * @brief Обучает SVM на загруженных данных и сохраняет модель.
     *
     * Перед вызовом необходимо загрузить данные через loadCSV().
     * Если model_path не пустой, обученная модель записывается в этот файл
     * (существующий файл перезаписывается).
     *
     * @return true при успешном обучении.
     */
    bool train();

    /**
     * @brief Предсказывает метку класса для одной точки признаков.
     * @param features Вектор признаков (T1, T2, T3).
     * @return +1 или -1. При ошибке возвращает 0.
     */
    int predict(const cv::Vec3f& features) const;

    /**
     * @brief Пакетное предсказание меток для набора точек.
     * @param samples Матрица признаков (N x 3, CV_32F).
     * @return Вектор меток (+1 или -1) размером N. Пустой при ошибке.
     */
    std::vector<int> predict(const cv::Mat& samples) const;

    /**
     * @brief Возвращает значение decision function (расстояние до гиперплоскости).
     *
     * Знак зависит от того, как была получена модель:
     *   - после train() знак выровнен через detectDecisionSignFlip():
     *     положительное значение соответствует классу +1;
     *   - после loadModel() знак не выравнивается (нет обучающих данных)
     *     и остаётся как в OpenCV: положительное значение соответствует
     *     первой метке в модели, для models/svm_model.xml это -1.
     *
     * Модуль значения от этого не зависит.
     *
     * @param features Вектор признаков (T1, T2, T3).
     * @return Знаковое расстояние до разделяющей гиперплоскости. 0.0f при ошибке.
     */
    float decisionFunction(const cv::Vec3f& features) const;

    bool isTrained() const noexcept { return is_trained_; }

    void setModelPath(const std::string& path) { model_path_ = path; }

private:
    cv::Ptr<cv::ml::SVM> svm_;
    cv::Mat              training_features_; ///< N x 3, CV_32F
    cv::Mat              training_labels_;   ///< N x 1, CV_32S (значения +1/-1)
    std::string          model_path_;
    bool                 is_trained_ = false;
    bool                 decision_sign_flipped_ = false;

    /**
     * @brief Определяет, нужно ли инвертировать знак decision function.
     *
     * cv::ml::SVM не гарантирует, что положительное значение RAW_OUTPUT
     * соответствует классу +1. Метод проверяет это на обучающих данных
     * и устанавливает флаг decision_sign_flipped_.
     *
     * Вызывается после train() и loadModel(). После loadModel() без
     * предшествующего loadCSV() обучающих данных нет, и флаг не меняется.
     *
     * @return true при успешном определении знака.
     */
    bool detectDecisionSignFlip();

    /**
     * @brief Вспомогательный метод: формирует cv::Mat 1x3 из Vec3f.
     */
    cv::Mat makeFeatureRow(const cv::Vec3f& features) const;


};