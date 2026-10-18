#pragma once

#include <opencv2/ml.hpp>
#include <string>

/**
 * @brief Обёртка над cv::ml::SVM для бинарной классификации пар точек черешка.
 *
 * Метки классов: +1 (черешок) и -1 (не черешок).
 * Признаки: (T1, T2, T3) — вещественные числа в диапазоне [0, 1].
 *
 * Порядок использования:
 *   1. Создать объект с путём к модели.
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
     * Метка "1" означает класс +1, любая другая — класс -1.
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
     * @brief Предсказывает метку класса для признаков (T1, T2, T3).
     * @return +1 или -1. Если модель не загружена и не обучена — 0.
     */
    int predict(const cv::Vec3f& features) const;

    /**
     * @brief Возвращает значение decision function (расстояние до гиперплоскости).
     *
     * Знак значения задаёт OpenCV: положительное значение соответствует
     * первой метке в модели, для models/svm_model.xml это -1.
     * Поэтому значение используется только по модулю: чем больше модуль,
     * тем дальше точка от границы классов.
     *
     * @return Значение decision function. Если модель не загружена и не обучена — 0.
     */
    float decisionFunction(const cv::Vec3f& features) const;

    void setModelPath(const std::string& path) { model_path_ = path; }

private:
    cv::Ptr<cv::ml::SVM> svm_;
    cv::Mat              training_features_; ///< N x 3, CV_32F
    cv::Mat              training_labels_;   ///< N x 1, CV_32S (значения +1/-1)
    std::string          model_path_;
    bool                 is_trained_ = false;

    /**
     * @brief Формирует cv::Mat 1x3 из признаков.
     */
    cv::Mat makeFeatureRow(const cv::Vec3f& features) const;
};