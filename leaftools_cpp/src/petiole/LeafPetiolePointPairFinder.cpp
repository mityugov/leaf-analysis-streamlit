#include "./LeafPetiolePointPairFinder.h"
#include "./EuclideanCND.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>

LeafPetiolePointPairFinder::LeafPetiolePointPairFinder(
    const cv::Mat& binary_image,
    const std::string& svm_model_path,
    const std::string& svm_csv_path,
    const Thresholds& thresholds)
    : binary_image_(binary_image)
    , geodesic_(0.0)
    , thresholds_(thresholds)
{
    findMaxContour();

    if (contour_.empty())
        return;

    gdc_ = std::make_unique<ContourGeodesicDistances>(
        std::make_unique<EuclideanCND>(contour_));

    total_length_ = gdc_->getTotalLength();
    geodesic_ = GeodesicParams(total_length_);

    initSVM(svm_model_path, svm_csv_path);
}

void LeafPetiolePointPairFinder::findMaxContour()
{
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary_image_, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    if (contours.empty())
        return;

    auto it = std::max_element(contours.begin(), contours.end(),
        [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b)
        {
            return cv::contourArea(a) < cv::contourArea(b);
        });

    contour_ = std::move(*it);
}

void LeafPetiolePointPairFinder::initSVM(const std::string& model_path,
    const std::string& csv_path)
{
    if (!model_path.empty() && svm_.loadModel(model_path))
        return;

    if (!csv_path.empty())
    {
        svm_.setModelPath(model_path);
        svm_.loadCSV(csv_path);
        svm_.train();
    }
}

bool LeafPetiolePointPairFinder::isValidPoint(const cv::Point& pt) const noexcept
{
    return pt.x >= 0 && pt.x < binary_image_.cols
        && pt.y >= 0 && pt.y < binary_image_.rows;
}

bool LeafPetiolePointPairFinder::isValidIndices(int i, int j) const noexcept
{
    const int n = static_cast<int>(contour_.size());
    return i >= 0 && j >= 0 && i < n && j < n && i != j;
}

bool LeafPetiolePointPairFinder::checkWhiteBetween() const
{
    // Середина отрезка AB
    cv::Point mid_point_AB((leaf_points_.ptA.x + leaf_points_.ptB.x) / 2, (leaf_points_.ptA.y + leaf_points_.ptB.y) / 2);
    if (!isValidPoint(mid_point_AB) || binary_image_.at<uchar>(mid_point_AB.y, mid_point_AB.x) == 0) {
        return false;
    }

    // Середина между A и серединой AB
    cv::Point mid_A((leaf_points_.ptA.x + mid_point_AB.x) / 2, (leaf_points_.ptA.y + mid_point_AB.y) / 2);
    if (!isValidPoint(mid_A) || binary_image_.at<uchar>(mid_A.y, mid_A.x) == 0) {
        return false;
    }

    // Середина между серединой AB и B
    cv::Point mid_B((mid_point_AB.x + leaf_points_.ptB.x) / 2, (mid_point_AB.y + leaf_points_.ptB.y) / 2);
    if (!isValidPoint(mid_B) || binary_image_.at<uchar>(mid_B.y, mid_B.x) == 0) {
        return false;
    }

    return true;
}

bool LeafPetiolePointPairFinder::checkBorderPixels() const noexcept
{
    // Вектор от A к B, нормированный
    const cv::Point2d dir_AB(
        leaf_points_.ptB.x - leaf_points_.ptA.x,
        leaf_points_.ptB.y - leaf_points_.ptA.y);
    const double len = cv::norm(dir_AB);
    if (len < thresholds_.distance_epsilon)
        return false;

    const cv::Point2d unit(dir_AB.x / len, dir_AB.y / len);

    const double offset = thresholds_.border_offset;

    // Точка на отрезке AB в offset пикселях от A
    const cv::Point neighborA(
        static_cast<int>(std::round(leaf_points_.ptA.x + unit.x * offset)),
        static_cast<int>(std::round(leaf_points_.ptA.y + unit.y * offset)));

    // Точка на отрезке AB в offset пикселях от B
    const cv::Point neighborB(
        static_cast<int>(std::round(leaf_points_.ptB.x - unit.x * offset)),
        static_cast<int>(std::round(leaf_points_.ptB.y - unit.y * offset)));

    // Оба соседних пикселя должны быть белыми (внутри листа)
    if (!isValidPoint(neighborA) || binary_image_.at<uchar>(neighborA.y, neighborA.x) == 0)
        return false;
    if (!isValidPoint(neighborB) || binary_image_.at<uchar>(neighborB.y, neighborB.x) == 0)
        return false;

    return true;
}

RatioCheck LeafPetiolePointPairFinder::checkGeodesicRatio(double low_threshold, double high_threshold) const noexcept
{
    const double ratio = geodesic_.getRatio();

    if (ratio <= low_threshold)
        return RatioCheck::TooSmall;
    if (ratio >= high_threshold)
        return RatioCheck::TooLarge;
    return RatioCheck::OK;
}

bool LeafPetiolePointPairFinder::checkMinWidthStem() const noexcept
{
    return (distances_.D_ab + distances_.D_cd) > thresholds_.width_threshold;
}

bool LeafPetiolePointPairFinder::checkStemFlatness() const noexcept
{
    const double D_as = cv::norm(leaf_points_.ptA - leaf_points_.ptS);
    const double D_bs = cv::norm(leaf_points_.ptB - leaf_points_.ptS);
    return distances_.D_ab <= std::max(D_as, D_bs);
}

bool LeafPetiolePointPairFinder::checkTPoint() const
{
    // Точка T лежит на продолжении отрезка AS за точкой S
    const double offset = thresholds_.t_point_offset;
    double Tx = leaf_points_.ptS.x + (leaf_points_.ptS.x - leaf_points_.ptA.x) * offset;
    double Ty = leaf_points_.ptS.y + (leaf_points_.ptS.y - leaf_points_.ptA.y) * offset;

    if (Tx < 0) Tx = 0;
    if (Tx >= binary_image_.cols) Tx = binary_image_.cols - 1;

    if (Ty < 0) Ty = 0;
    if (Ty >= binary_image_.rows) Ty = binary_image_.rows - 1;

    int x_idx = static_cast<int>(std::ceil(Tx));
    int y_idx = static_cast<int>(std::ceil(Ty));

    if (binary_image_.at<uchar>(y_idx, x_idx) == 255)
        return false;
    return true;
}

StemKeyDistances LeafPetiolePointPairFinder::computeStemKeyDistances(
    size_t i, size_t j) const
{
    StemKeyDistances result;
    const double half = geodesic_.getGMin() / 2.0;
    const double quart = geodesic_.getGMin() / 4.0;

    // Вычисляем базовое геодезическое расстояние от 0 до j
    // с учётом возможного перехода через начало контура (j < i).
    double G_base;
    if (j >= i)
    {
        G_base = gdc_->getGeodesicDistance(0, j);
    }
    else
    {
        // Путь от 0 до j проходит через конец контура
        G_base = total_length_ + gdc_->getGeodesicDistance(0, j);
    }

    // Опорные геодезические расстояния:
    //   G_k — середина дуги i → j;
    //   G_c — 1/4 дуги от i (ближе к i);
    //   G_d — 3/4 дуги от i (ближе к j).
    result.G_k = G_base - half;
    result.G_c = G_base - 3.0 * quart;
    result.G_d = G_base - quart;

    // Нормализуем в [0, total_length_)
    auto wrap = [&](double g) -> double
        {
            if (g < 0.0)         g += total_length_;
            if (g > total_length_) g -= total_length_;
            return g;
        };

    result.G_k = wrap(result.G_k);
    result.G_c = wrap(result.G_c);
    result.G_d = wrap(result.G_d);

    return result;
}

double LeafPetiolePointPairFinder::phi(double D) const noexcept
{
    return thresholds_.alpha * std::pow(1.15, D - std::max(0.0, 10 * (1.1 * distances_.De - D)))
        + (1 - thresholds_.alpha) * std::log(std::max(1.0, D)) / std::log(1.5);
}

T_Parameters LeafPetiolePointPairFinder::computeTParameters() const noexcept
{
    T_Parameters tp;

    // T1: насколько phi(D_ab + штраф) вписывается в G_min
    const double corrected_Dab =
        distances_.D_ab + 2 * std::max(distances_.D_ab - distances_.D_cd, 0.0);
    tp.T1 = 1.0 - std::min(phi(corrected_Dab), geodesic_.getGMin()) / geodesic_.getGMin();

    // T2: отношение пути через S к геодезике + phi
    const cv::Point mid_AB(
        (leaf_points_.ptA.x + leaf_points_.ptB.x) / 2,
        (leaf_points_.ptA.y + leaf_points_.ptB.y) / 2);
    const cv::Point mid_CD(
        (leaf_points_.ptC.x + leaf_points_.ptD.x) / 2,
        (leaf_points_.ptC.y + leaf_points_.ptD.y) / 2);

    const double D_ab_cd = cv::norm(mid_AB - mid_CD);
    const double D_cd_s = cv::norm(mid_CD - leaf_points_.ptS);

    tp.T2 = (2.0 * (D_ab_cd + D_cd_s))
        / (geodesic_.getGMin() + distances_.D_ab + phi(std::abs(distances_.D_cd - distances_.D_ab)));

    // T3: доля G_min в суммарной длине
    tp.T3 = geodesic_.getGMin()
        / (geodesic_.getGMin() + geodesic_.getGMax() + distances_.D_ab);

    // Итоговый балл
    tp.T = thresholds_.w1 * tp.T1
        + thresholds_.w2 * tp.T2
        + thresholds_.w3 * 2.0 * tp.T3;

    return tp;
}

bool LeafPetiolePointPairFinder::svmPredict(const T_Parameters& tp) const
{
    return svm_.predict(cv::Vec3f(
        static_cast<float>(tp.T1),
        static_cast<float>(tp.T2),
        static_cast<float>(tp.T3))) == 1;
}

float LeafPetiolePointPairFinder::svmDecision(const T_Parameters& tp) const
{
    return svm_.decisionFunction(cv::Vec3f(
        static_cast<float>(tp.T1),
        static_cast<float>(tp.T2),
        static_cast<float>(tp.T3)));
}

bool LeafPetiolePointPairFinder::checkStemPoints(int i, int j)
{
    const auto skd = computeStemKeyDistances(i, j);

    const size_t k = gdc_->getIndexByDistance(skd.G_k);
    leaf_points_.ptS = contour_[k];

    if (!checkStemFlatness())
        return false;

    const size_t c = gdc_->getIndexByDistance(skd.G_c);
    const size_t d = gdc_->getIndexByDistance(skd.G_d);
    leaf_points_.ptC = contour_[c];
    leaf_points_.ptD = contour_[d];

    distances_.D_cd = cv::norm(leaf_points_.ptC - leaf_points_.ptD);
    if (distances_.D_cd < thresholds_.distance_epsilon)
        return false;

    if (!checkMinWidthStem())
        return false;

    if (!checkBorderPixels())
        return false;

    if (!checkTPoint())
        return false;

    return true;
}

LeafPetiolePointPairFinder::Candidate LeafPetiolePointPairFinder::runScanPass(
    const cv::Point* ref_ptS, double D_max)
{
    Candidate best;
    const int n = static_cast<int>(contour_.size());
    const int contour_step = n / max_scan_points + 1;

    for (int i = 0; i < n; i += contour_step)
    {
        for (int step = 1; step < n; step += contour_step)
        {
            int j = (i + step) % n;

            if (!isValidIndices(i, j))
                continue;

            // Расстояние AB
            leaf_points_.ptA = contour_[i];
            leaf_points_.ptB = contour_[j];
            distances_.D_ab = cv::norm(leaf_points_.ptA - leaf_points_.ptB);
            if (distances_.D_ab < thresholds_.distance_epsilon)
                continue;

            // Обновляем D_max независимо от остальных проверок
            distances_.D_max = std::max(distances_.D_max, distances_.D_ab);

            // Быстрые геодезические проверки
            geodesic_.set(gdc_->getGeodesicDistance(i, j));

            const RatioCheck rc = checkGeodesicRatio(thresholds_.low_threshold, thresholds_.high_threshold);
            if (rc == RatioCheck::TooLarge)
                continue;
            if (rc == RatioCheck::TooSmall)
                continue;

            // Проверка белых пикселей (дороже, поэтому после геодезических проверок)
            if (!checkWhiteBetween())
                continue;

            // Фильтрация для третьего прохода (дешевле, поэтому до остальных проверок)
            if (ref_ptS != nullptr)
            {
                const auto skd_check = computeStemKeyDistances(i, j);
                const cv::Point ptS_check = contour_[gdc_->getIndexByDistance(skd_check.G_k)];

                const double D_S1_S2 = cv::norm(*ref_ptS - ptS_check);
                const double rel_diff = std::abs(D_max - D_S1_S2);
                if (rel_diff > thresholds_.D_S_rel_tolerance * D_max)
                    continue;
            }

            if (!checkStemPoints(i, j))
                continue;

            // Метрики
            const T_Parameters tp = computeTParameters();

            if (!tp.isValid(thresholds_))
                continue;

            // Обновляем лучший кандидат
            if (tp.T > best.tp.T)
            {
                best.ptA = leaf_points_.ptA;
                best.ptB = leaf_points_.ptB;
                best.ptS = leaf_points_.ptS;
                best.tp = tp;
                best.valid = true;
            }
        }
    }

    return best;
}

FindResult LeafPetiolePointPairFinder::makeResult(const Candidate& candidate, bool found) const
{
    FindResult result;
    result.ptA = candidate.ptA;
    result.ptB = candidate.ptB;
    result.ptS = candidate.ptS;
    result.found = found;
    return result;
}

FindResult LeafPetiolePointPairFinder::findBestPair()
{
    if (contour_.empty())
    {
        std::cerr << "[LeafPetiolePointPairFinder] Контур пуст, поиск невозможен.\n";
        return {};
    }

    // Первый проход: ищем минимальное De (snake mode)
    {
        struct DeCandidate
        {
            double D_ab = std::numeric_limits<double>::max();
        };

        const int max_pct = static_cast<int>(thresholds_.snake_high_threshold * 100.0);
        const int min_pct = static_cast<int>(thresholds_.snake_low_threshold * 100.0);
        std::map<int, DeCandidate> candidatesByPct;
        for (int pct = min_pct; pct <= max_pct; ++pct)
            candidatesByPct[pct] = {};

        const int n = static_cast<int>(contour_.size());
        const int contour_step = n / max_scan_points + 1;
        for (int i = 0; i < n; i += contour_step)
        {
            for (int step = 1; step < n; step += contour_step)
            {
                const int j = (i + step) % n;

                if (!isValidIndices(i, j))
                    continue;

                leaf_points_.ptA = contour_[i];
                leaf_points_.ptB = contour_[j];

                if (!checkWhiteBetween())
                    continue;

                double G_min = gdc_->getGeodesicDistance(i, j);
                geodesic_.set(G_min);

                const RatioCheck rc = checkGeodesicRatio(thresholds_.snake_low_threshold, thresholds_.high_threshold);
                if (rc == RatioCheck::TooSmall)
                    continue;
                if (rc == RatioCheck::TooLarge)
                    break;

                const double ratio = geodesic_.getRatio();
                const int ratio_pct = std::round(ratio * 100.0);

                if (ratio_pct > max_pct)
                    break;

                if (ratio_pct < min_pct)
                    continue;

                distances_.D_ab = cv::norm(leaf_points_.ptA - leaf_points_.ptB);
                if (distances_.D_ab < thresholds_.distance_epsilon)
                    continue;

                if (!checkStemPoints(i, j))
                    continue;

                // Если D_ab меньше найденного ранее для этого процента,
                // запоминаем его и переходим к следующему i
                if (distances_.D_ab < candidatesByPct[ratio_pct].D_ab)
                {
                    candidatesByPct[ratio_pct].D_ab = distances_.D_ab;
                    break;
                }
            }
        }

        // De — первое D_ab меньше max_De в порядке возрастания процента
        bool deFound = false;
        for (const auto& [pct, candidate] : candidatesByPct)
        {
            if (candidate.D_ab >= thresholds_.max_De)
                continue;

            distances_.De = candidate.D_ab;
            deFound = true;
            break;
        }

        if (!deFound)
            return {};
    }

    // Второй проход: ищем кандидата с максимальным T
    Candidate cand1 = runScanPass();

    if (!cand1.valid)
        return {};

    // Ранний выход: T высокий — черешок найден без SVM
    if (cand1.tp.T >= thresholds_.T_accept_threshold)
        return makeResult(cand1, true);

    // Ранний выход: T низкий — черешок не найден, SVM не вызывается
    if (cand1.tp.T <= thresholds_.T_reject_threshold)
        return makeResult(cand1, false);

    // Дальше T лежит между T_reject_threshold и T_accept_threshold,
    // решение принимает SVM.

    // Третий проход: ищем альтернативного кандидата
    Candidate cand2 = runScanPass(
        /*ref_ptS=*/&cand1.ptS,
        /*D_max=*/distances_.D_max);

    // Принятие решения
    if (!cand2.valid)
    {
        // Только один кандидат — спрашиваем SVM
        return makeResult(cand1, svmPredict(cand1.tp));
    }

    const double dT = std::abs(cand1.tp.T - cand2.tp.T);

    if (dT >= thresholds_.T_epsilon)
    {
        // Кандидаты сильно различаются по T — спрашиваем SVM про кандидата с наибольшим T
        return makeResult(cand1, svmPredict(cand1.tp));
    }

    // Кандидаты близки по T — используем SVM для обоих
    const bool p1 = svmPredict(cand1.tp);
    const bool p2 = svmPredict(cand2.tp);

    if (p1 && !p2)
        return makeResult(cand1, true);
    if (!p1 && p2)
        return makeResult(cand2, true);
    if (!p1 && !p2)
        return {}; // Оба отклонены

    // Оба приняты SVM — выбираем кандидата с большим |decision function|,
    // то есть более удалённого от границы классов
    const float d1 = svmDecision(cand1.tp);
    const float d2 = svmDecision(cand2.tp);
    const Candidate& winner = (std::abs(d1) >= std::abs(d2)) ? cand1 : cand2;
    return makeResult(winner, true);
}