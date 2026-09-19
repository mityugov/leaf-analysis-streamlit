#include "./SVM_Classifier.h"

#include <fstream>
#include <iostream>
#include <sstream>

// ─────────────────────────────────────────────────────────────────────────────
// Конструктор
// ─────────────────────────────────────────────────────────────────────────────

SVM_Classifier::SVM_Classifier(const std::string& model_path)
    : model_path_(model_path)
{
}

// ─────────────────────────────────────────────────────────────────────────────
// Вспомогательный метод
// ─────────────────────────────────────────────────────────────────────────────

cv::Mat SVM_Classifier::makeFeatureRow(const cv::Vec3f& features) const
{
    cv::Mat sample(1, 3, CV_32F);
    sample.at<float>(0, 0) = features[0];
    sample.at<float>(0, 1) = features[1];
    sample.at<float>(0, 2) = features[2];
    return sample;
}

// ─────────────────────────────────────────────────────────────────────────────
// Загрузка CSV
// ─────────────────────────────────────────────────────────────────────────────

bool SVM_Classifier::loadCSV(const std::string& csv_path)
{
    std::ifstream file(csv_path);
    if (!file.is_open())
    {
        std::cerr << "[SVM_Classifier] Не удалось открыть файл: " << csv_path << '\n';
        return false;
    }

    auto toFloat = [](const std::string& s) -> float
        {
            std::istringstream iss(s);
            iss.imbue(std::locale::classic());
            float val = 0.0f;
            iss >> val;
            return val;
        };

    std::vector<float> t1_vec, t2_vec, t3_vec;
    std::vector<int>   labels;
    std::string        line;

    // Пропускаем заголовок
    std::getline(file, line);

    int line_number = 1;
    while (std::getline(file, line))
    {
        ++line_number;

        if (line.empty())
            continue;

        std::stringstream        ss(line);
        std::string              token;
        std::vector<std::string> row;
        row.reserve(4);

        while (std::getline(ss, token, ';'))
            row.push_back(token);

        if (row.size() < 4)
        {
            std::cerr << "[SVM_Classifier] Строка " << line_number
                << ": недостаточно полей, пропускаем: " << line << '\n';
            continue;
        }

        try
        {
            // Метка: только "1" → +1, всё остальное → -1
            const int label = (row[0] == "1") ? 1 : -1;
            labels.push_back(label);
            t1_vec.push_back(toFloat(row[1]));
            t2_vec.push_back(toFloat(row[2]));
            t3_vec.push_back(toFloat(row[3]));
        }
        catch (const std::exception& e)
        {
            std::cerr << "[SVM_Classifier] Строка " << line_number
                << ": ошибка парсинга (" << e.what() << "), пропускаем.\n";
        }
    }

    if (t1_vec.empty())
    {
        std::cerr << "[SVM_Classifier] Файл не содержит валидных данных: " << csv_path << '\n';
        return false;
    }

    const int n = static_cast<int>(t1_vec.size());
    training_features_ = cv::Mat(n, 3, CV_32F);
    training_labels_ = cv::Mat(n, 1, CV_32S);

    for (int i = 0; i < n; ++i)
    {
        training_features_.at<float>(i, 0) = t1_vec[i];
        training_features_.at<float>(i, 1) = t2_vec[i];
        training_features_.at<float>(i, 2) = t3_vec[i];
        training_labels_.at<int>(i, 0) = labels[i];
    }

    std::cout << "[SVM_Classifier] Загружено " << n
        << " примеров из " << csv_path << '\n';
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Обучение
// ─────────────────────────────────────────────────────────────────────────────

bool SVM_Classifier::train()
{
    if (training_features_.empty())
    {
        std::cerr << "[SVM_Classifier] Нет данных для обучения. "
            "Сначала вызовите loadCSV().\n";
        return false;
    }

    auto train_data = cv::ml::TrainData::create(
        training_features_,
        cv::ml::ROW_SAMPLE,
        training_labels_);

    svm_ = cv::ml::SVM::create();
    svm_->setType(cv::ml::SVM::C_SVC);
    svm_->setKernel(cv::ml::SVM::LINEAR);
    svm_->setC(100.0);
    svm_->setTermCriteria(cv::TermCriteria(
        cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS,
        100'000, 1e-6));

    if (!svm_->train(train_data))
    {
        std::cerr << "[SVM_Classifier] Обучение завершилось неудачей.\n";
        return false;
    }

    is_trained_ = true;
    std::cout << "[SVM_Classifier] Модель обучена. "
        << "Число опорных векторов: " << svm_->getSupportVectors().rows << '\n';

    if (!model_path_.empty())
    {
        svm_->save(model_path_);
        std::cout << "[SVM_Classifier] Модель сохранена: " << model_path_ << '\n';
    }

    detectDecisionSignFlip();
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Загрузка модели
// ─────────────────────────────────────────────────────────────────────────────

bool SVM_Classifier::loadModel(const std::string& model_path)
{
    const std::string& path = model_path.empty() ? model_path_ : model_path;

    if (path.empty())
    {
        std::cerr << "[SVM_Classifier] Путь к модели не задан.\n";
        return false;
    }

    try
    {
        svm_ = cv::ml::SVM::load(path);
        is_trained_ = true;
        std::cout << "[SVM_Classifier] Модель загружена: " << path << '\n';
        detectDecisionSignFlip();
        return true;
    }
    catch (const cv::Exception& e)
    {
        std::cerr << "[SVM_Classifier] Ошибка загрузки модели (OpenCV): "
            << e.what() << '\n';
    }
    catch (const std::exception& e)
    {
        std::cerr << "[SVM_Classifier] Ошибка загрузки модели: "
            << e.what() << '\n';
    }
    return false;
}

// ─────────────────────────────────────────────────────────────────────────────
// Предсказание
// ─────────────────────────────────────────────────────────────────────────────

int SVM_Classifier::predict(const cv::Vec3f& features) const
{
    if (!is_trained_)
    {
        std::cerr << "[SVM_Classifier] Модель не обучена.\n";
        return 0;
    }

    cv::Mat result;
    svm_->predict(makeFeatureRow(features), result);
    return static_cast<int>(result.at<float>(0, 0));
}

std::vector<int> SVM_Classifier::predict(const cv::Mat& samples) const
{
    if (!is_trained_)
    {
        std::cerr << "[SVM_Classifier] Модель не обучена.\n";
        return {};
    }

    if (samples.cols != 3 || samples.type() != CV_32F)
    {
        std::cerr << "[SVM_Classifier] predict(): ожидается матрица N x 3 CV_32F.\n";
        return {};
    }

    cv::Mat results;
    svm_->predict(samples, results);

    std::vector<int> predictions;
    predictions.reserve(results.rows);
    for (int i = 0; i < results.rows; ++i)
        predictions.push_back(static_cast<int>(results.at<float>(i, 0)));

    return predictions;
}

// ─────────────────────────────────────────────────────────────────────────────
// Decision function
// ─────────────────────────────────────────────────────────────────────────────

float SVM_Classifier::decisionFunction(const cv::Vec3f& features) const
{
    if (!is_trained_)
    {
        std::cerr << "[SVM_Classifier] Модель не обучена.\n";
        return 0.0f;
    }

    float score = svm_->predict(
        makeFeatureRow(features),
        cv::noArray(),
        cv::ml::StatModel::RAW_OUTPUT);

    return decision_sign_flipped_ ? -score : score;
}

// ─────────────────────────────────────────────────────────────────────────────
// Определение инверсии знака decision function
// ─────────────────────────────────────────────────────────────────────────────

bool SVM_Classifier::detectDecisionSignFlip()
{
    // Для определения знака используем обучающие данные, если они есть.
    // Если модель загружена без CSV (loadModel без предшествующего loadCSV),
    // проверка невозможна — оставляем decision_sign_flipped_ = false.
    if (!is_trained_ || training_features_.empty() || training_labels_.empty())
        return false;

    double sum_pos = 0.0, sum_neg = 0.0;
    int    cnt_pos = 0, cnt_neg = 0;

    for (int i = 0; i < training_features_.rows; ++i)
    {
        const float score = svm_->predict(
            training_features_.row(i),
            cv::noArray(),
            cv::ml::StatModel::RAW_OUTPUT);

        if (training_labels_.at<int>(i, 0) == 1)
        {
            sum_pos += score;
            ++cnt_pos;
        }
        else
        {
            sum_neg += score;
            ++cnt_neg;
        }
    }

    if (cnt_pos == 0 || cnt_neg == 0)
    {
        std::cerr << "[SVM_Classifier] detectDecisionSignFlip: "
            "один из классов отсутствует в обучающих данных.\n";
        return false;
    }

    // Если среднее значение RAW_OUTPUT у класса +1 меньше, чем у -1,
    // знак нужно инвертировать.
    decision_sign_flipped_ = (sum_pos / cnt_pos) < (sum_neg / cnt_neg);
    return true;
}