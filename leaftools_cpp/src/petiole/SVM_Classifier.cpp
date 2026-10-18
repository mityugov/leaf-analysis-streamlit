#include "./SVM_Classifier.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

SVM_Classifier::SVM_Classifier(const std::string& model_path)
    : model_path_(model_path)
{
}

cv::Mat SVM_Classifier::makeFeatureRow(const cv::Vec3f& features) const
{
    cv::Mat sample(1, 3, CV_32F);
    sample.at<float>(0, 0) = features[0];
    sample.at<float>(0, 1) = features[1];
    sample.at<float>(0, 2) = features[2];
    return sample;
}

bool SVM_Classifier::loadCSV(const std::string& csv_path)
{
    std::ifstream file(csv_path);
    if (!file.is_open())
    {
        std::cerr << "[SVM_Classifier] Не удалось открыть файл: " << csv_path << '\n';
        return false;
    }

    // Число с точкой в качестве разделителя, независимо от локали системы
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

        while (std::getline(ss, token, ';'))
            row.push_back(token);

        if (row.size() < 4)
        {
            std::cerr << "[SVM_Classifier] Строка " << line_number
                << ": недостаточно полей, пропускаем: " << line << '\n';
            continue;
        }

        if (row[0] == "1")
            labels.push_back(1);
        else
            labels.push_back(-1);

        t1_vec.push_back(toFloat(row[1]));
        t2_vec.push_back(toFloat(row[2]));
        t3_vec.push_back(toFloat(row[3]));
    }

    if (t1_vec.empty())
    {
        std::cerr << "[SVM_Classifier] Файл не содержит данных: " << csv_path << '\n';
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

    return true;
}

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

    if (!model_path_.empty())
        svm_->save(model_path_);

    return true;
}

bool SVM_Classifier::loadModel(const std::string& model_path)
{
    std::string path = model_path;
    if (path.empty())
        path = model_path_;

    if (path.empty())
    {
        std::cerr << "[SVM_Classifier] Путь к модели не задан.\n";
        return false;
    }

    // Если файла нет или он повреждён, OpenCV бросает cv::Exception
    try
    {
        svm_ = cv::ml::SVM::load(path);
    }
    catch (const cv::Exception& e)
    {
        std::cerr << "[SVM_Classifier] Ошибка загрузки модели: " << e.what() << '\n';
        return false;
    }

    if (svm_.empty())
    {
        std::cerr << "[SVM_Classifier] Не удалось загрузить модель: " << path << '\n';
        return false;
    }

    is_trained_ = true;
    return true;
}

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

float SVM_Classifier::decisionFunction(const cv::Vec3f& features) const
{
    if (!is_trained_)
    {
        std::cerr << "[SVM_Classifier] Модель не обучена.\n";
        return 0.0f;
    }

    return svm_->predict(
        makeFeatureRow(features),
        cv::noArray(),
        cv::ml::StatModel::RAW_OUTPUT);
}