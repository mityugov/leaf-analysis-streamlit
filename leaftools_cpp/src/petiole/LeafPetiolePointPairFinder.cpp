#include "./LeafPetiolePointPairFinder.h"

#include <chrono>
#include <iostream>

using namespace std;
using namespace cv;

// ─────────────────────────────────────────────────────────────────────────────
// Конструктор
// ─────────────────────────────────────────────────────────────────────────────

LeafPetiolePointPairFinder::LeafPetiolePointPairFinder(
    const cv::Mat& binary_image,
    const std::string& svm_model_path,
    const std::string& svm_csv_path,
    const Thresholds& thresholds) 
    : binary_image_(binary_image)
    , thresholds_(thresholds)
    , geodesic_(0.0)
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

// ─────────────────────────────────────────────────────────────────────────────
// Инициализация
// ─────────────────────────────────────────────────────────────────────────────

void LeafPetiolePointPairFinder::findMaxContour()
{
    vector<vector<Point>> contours;
    findContours(binary_image_, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

    if (contours.empty())
        return;

    auto it = max_element(contours.begin(), contours.end(),
        [](const vector<Point>& a, const vector<Point>& b)
        {
            return contourArea(a) < contourArea(b);
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

// ─────────────────────────────────────────────────────────────────────────────
// Валидация
// ─────────────────────────────────────────────────────────────────────────────

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

// ─────────────────────────────────────────────────────────────────────────────
// Проверки (constraints)
// ─────────────────────────────────────────────────────────────────────────────

bool LeafPetiolePointPairFinder::checkWhiteBetween() const
{
    ////// Используем LineIterator для обхода всех пикселей отрезка A-B,
    ////// а не только трёх опорных точек — это исключает false positive
    ////// на вогнутых объектах и наклонных отрезках.
    //cv::LineIterator it(binary_image_, leaf_points_.ptA, leaf_points_.ptB, 8);
    //for (int i = 0; i < it.count; ++i, ++it)
    //{
    //    if (binary_image_.at<uchar>(it.pos()) == 0)
    //        return false;
    //}
    //return true;

    // Check direct mid point
    cv::Point mid_point_AB((leaf_points_.ptA.x + leaf_points_.ptB.x) / 2, (leaf_points_.ptA.y + leaf_points_.ptB.y) / 2);
    if (!isValidPoint(mid_point_AB) || binary_image_.at<uchar>(mid_point_AB.y, mid_point_AB.x) == 0) {
        return false;
    }

    // Check mid point between A and mid_point_AB
    cv::Point mid_A((leaf_points_.ptA.x + mid_point_AB.x) / 2, (leaf_points_.ptA.y + mid_point_AB.y) / 2);
    if (!isValidPoint(mid_A) || binary_image_.at<uchar>(mid_A.y, mid_A.x) == 0) {
        return false;
    }

    // Check mid point between mid_point_AB and B
    cv::Point mid_B((mid_point_AB.x + leaf_points_.ptB.x) / 2, (mid_point_AB.y + leaf_points_.ptB.y) / 2);
    if (!isValidPoint(mid_B) || binary_image_.at<uchar>(mid_B.y, mid_B.x) == 0) {
        return false;
    }

    //return true;
}

bool LeafPetiolePointPairFinder::checkBorderPixels() const noexcept
{
    ////// Используем LineIterator для обхода всех пикселей отрезка A-B,
    ////// а не только трёх опорных точек — это исключает false positive
    ////// на вогнутых объектах и наклонных отрезках.
    //cv::LineIterator it(binary_image_, leaf_points_.ptA, leaf_points_.ptB, 8);
    //for (int i = 0; i < it.count; ++i, ++it)
    //{
    //    if (binary_image_.at<uchar>(it.pos()) == 0)
    //        return false;
    //}
    //return true;
 
    // Вектор от A к B, нормированный
    const cv::Point2d dir_AB(
        leaf_points_.ptB.x - leaf_points_.ptA.x,
        leaf_points_.ptB.y - leaf_points_.ptA.y);
    const double len = cv::norm(dir_AB);
    if (len < thresholds_.distance_epsilon)
        return false;

    const cv::Point2d unit(dir_AB.x / len, dir_AB.y / len);

    double coef = 2;
    const cv::Point neighborA(
        static_cast<int>(std::round(leaf_points_.ptA.x + unit.x * coef)),
        static_cast<int>(std::round(leaf_points_.ptA.y + unit.y * coef)));

    // Сосед точки B по направлению к A (1 пиксель внутрь)
    const cv::Point neighborB(
        static_cast<int>(std::round(leaf_points_.ptB.x - unit.x * coef)),
        static_cast<int>(std::round(leaf_points_.ptB.y - unit.y * coef)));

    // Оба соседних пикселя должны быть белыми (внутри листа)
    if (!isValidPoint(neighborA) || binary_image_.at<uchar>(neighborA.y, neighborA.x) == 0)
        return false;
    if (!isValidPoint(neighborB) || binary_image_.at<uchar>(neighborB.y, neighborB.x) == 0)
        return false;

    return true;
}

RatioCheck LeafPetiolePointPairFinder::checkGeodesicRatio(double low_threshold, double high_threshold) const noexcept
{
    if (high_threshold < 0.0)
        high_threshold = thresholds_.high_threshold;

    if (low_threshold < 0.0)
        low_threshold = thresholds_.low_threshold;

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

bool LeafPetiolePointPairFinder::checkTPoint()
{
    double Tx = leaf_points_.ptS.x + (leaf_points_.ptS.x - leaf_points_.ptA.x) * 0.05;
    double Ty = leaf_points_.ptS.y + (leaf_points_.ptS.y - leaf_points_.ptA.y) * 0.05;

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

// ─────────────────────────────────────────────────────────────────────────────
// Вычисление геодезических расстояний до опорных точек
// ─────────────────────────────────────────────────────────────────────────────

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
    //   G_k = середина дуги [i..j]
    //   G_c = 3/4 от i к j (ближе к j)
    //   G_d = 1/4 от i к j (ближе к i)
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

// ─────────────────────────────────────────────────────────────────────────────
// Метрики T1, T2, T3
// ─────────────────────────────────────────────────────────────────────────────

double LeafPetiolePointPairFinder::phi(double D) const noexcept
{
    return thresholds_.alpha * pow(1.15, D - std::max(0.0, 10 * (1.1 * distances_.De - D)))
        + (1 - thresholds_.alpha) * log(std::max(1.0, D)) / log(1.5);
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

    tp.D_ab = distances_.D_ab;
    tp.D_cd = distances_.D_cd;

    return tp;
}

// ─────────────────────────────────────────────────────────────────────────────
// SVM helpers
// ─────────────────────────────────────────────────────────────────────────────

bool LeafPetiolePointPairFinder::svmPredict(const T_Parameters& tp) const
{
    return svm_.predict(cv::Vec3f(
        static_cast<float>(tp.T1),
        static_cast<float>(tp.T2),
        static_cast<float>(tp.T3))) == 1.0f;
}

float LeafPetiolePointPairFinder::svmDecision(const T_Parameters& tp)
{
    return svm_.decisionFunction(cv::Vec3f(
        static_cast<float>(tp.T1),
        static_cast<float>(tp.T2),
        static_cast<float>(tp.T3)));
}

// ─────────────────────────────────────────────────────────────────────────────
// analyzePairPoints
// ─────────────────────────────────────────────────────────────────────────────

ErrorCode LeafPetiolePointPairFinder::analyzePairPoints(int i, int j)
{
    if (!isValidIndices(i, j))
        return ErrorCode::INVALID_INDEXES;

    leaf_points_.ptA = contour_[i];
    leaf_points_.ptB = contour_[j];
    distances_.D_ab = cv::norm(leaf_points_.ptA - leaf_points_.ptB);

    if (distances_.D_ab < thresholds_.distance_epsilon)
        return ErrorCode::ZERO_DISTANCE_AB;

    geodesic_.set(gdc_->getGeodesicDistance(i, j));

    if (checkGeodesicRatio(thresholds_.low_threshold, thresholds_.high_threshold) == RatioCheck::TooSmall)
        return ErrorCode::MIN_GEODESIC_RATIO_INVALID;
    if (checkGeodesicRatio(thresholds_.low_threshold, thresholds_.high_threshold) == RatioCheck::TooLarge)
        return ErrorCode::MAX_GEODESIC_RATIO_INVALID;

    if (!checkWhiteBetween())
        return ErrorCode::NO_WHITE_BETWEEN;

    if (!checkBorderPixels())
        return ErrorCode::NO_WHITE_BETWEEN;

    const auto skd = computeStemKeyDistances(i, j);
    const size_t k = gdc_->getIndexByDistance(skd.G_k);
    leaf_points_.ptS = contour_[k];

    if (!checkStemFlatness())
        return ErrorCode::STEM_FLATNESS_FAIL;

    const size_t c = gdc_->getIndexByDistance(skd.G_c);
    const size_t d = gdc_->getIndexByDistance(skd.G_d);
    leaf_points_.ptC = contour_[c];
    leaf_points_.ptD = contour_[d];

    distances_.D_cd = cv::norm(leaf_points_.ptC - leaf_points_.ptD);
    if (distances_.D_cd < thresholds_.distance_epsilon)
        return ErrorCode::ZERO_DISTANCE_CD;

    if (!checkMinWidthStem())
        return ErrorCode::MIN_WIDTH_STEM_FAIL;

    tparameters_ = computeTParameters();

    if (!tparameters_.isValid(thresholds_))
        return ErrorCode::INVALID_T_PARAMETERS;

    return ErrorCode::SUCCESS;
}

// ─────────────────────────────────────────────────────────────────────────────
// Единый проход O(n²)
// ─────────────────────────────────────────────────────────────────────────────

LeafPetiolePointPairFinder::Candidate LeafPetiolePointPairFinder::runScanPass(
    const cv::Point* ref_ptS, double D_max)
{
    Candidate best;
    const int n = static_cast<int>(contour_.size());
    const int contour_step = (n / 10000) + 1;
    //const int contour_step = 1;

    for (int i = 0; i < n; i += contour_step)
    {
        //if (i != 5904)
        //    continue;

        for (int step = 1; step < n; step += contour_step)
        {
            int j = (i + step) % n;

            //if (j != 6310)
            //    continue;

            if (!isValidIndices(i, j))
                continue;

            // ── Расстояние AB ─────────────────────────────────────────────────
            leaf_points_.ptA = contour_[i];
            leaf_points_.ptB = contour_[j];
            distances_.D_ab = cv::norm(leaf_points_.ptA - leaf_points_.ptB);
   

            if (distances_.D_ab < thresholds_.distance_epsilon)
                continue;

            // Обновляем D_max независимо от остальных проверок
            distances_.D_max = std::max(distances_.D_max, distances_.D_ab);

            // ── Быстрые геодезические проверки ───────────────────────────────
            geodesic_.set(gdc_->getGeodesicDistance(i, j));

            const RatioCheck rc = checkGeodesicRatio(thresholds_.low_threshold, thresholds_.high_threshold);
            if (rc == RatioCheck::TooLarge)
                continue;
            if (rc == RatioCheck::TooSmall)
                continue;

            // ── Проверка белых пикселей (дороже — делаем позже) ───────────────
            if (!checkWhiteBetween())
                continue;

            // ── Фильтрация для третьего прохода (дёшево — до остальных) ──────
            if (ref_ptS != nullptr)
            {
                const auto skd_check = computeStemKeyDistances(i, j);
                const cv::Point ptS_check = contour_[gdc_->getIndexByDistance(skd_check.G_k)];

                const double D_S1_S2 = cv::norm(*ref_ptS - ptS_check);
                const double rel_diff = std::abs(D_max - D_S1_S2);
                if (rel_diff > thresholds_.D_S_rel_tolerance * D_max)
                    continue;
            }

            // ── Опорные точки стебля ──────────────────────────────────────────
            const auto skd = computeStemKeyDistances(i, j);

            const size_t k = gdc_->getIndexByDistance(skd.G_k);
            leaf_points_.ptS = contour_[k];

            if (!checkStemFlatness())
                continue;

            const size_t c = gdc_->getIndexByDistance(skd.G_c);
            const size_t d = gdc_->getIndexByDistance(skd.G_d);
            leaf_points_.ptC = contour_[c];
            leaf_points_.ptD = contour_[d];

            distances_.D_cd = cv::norm(leaf_points_.ptC - leaf_points_.ptD);
            if (distances_.D_cd < thresholds_.distance_epsilon)
                continue;

            if (!checkMinWidthStem())
                continue;

            if (!checkBorderPixels())
                continue;

            if (!checkTPoint())
                continue;

            // ── Метрики ───────────────────────────────────────────────────────
            const T_Parameters tp = computeTParameters();

            if (!tp.isValid(thresholds_))
                continue;

            // ── Обновляем лучший кандидат ─────────────────────────────────────
            if (tp.T > best.tp.T)
            {
                best.ptA = leaf_points_.ptA;
                best.ptB = leaf_points_.ptB;
                best.ptS = leaf_points_.ptS;
                best.tp = tp;
                best.D_ab = distances_.D_ab;
                best.idx_i = i;
                best.idx_j = j;
                best.valid = true;
            }
        }
    }

    return best;
}

// ─────────────────────────────────────────────────────────────────────────────
// findBestPair  (заменяет compareTAndFindD)
// ─────────────────────────────────────────────────────────────────────────────

FindResult LeafPetiolePointPairFinder::findBestPair()
{
    if (contour_.empty())
    {
        std::cerr << "[LeafPetiolePointPairFinder] Контур пуст, поиск невозможен.\n";
        return {};
    }

    std::cout << "[LeafPetiolePointPairFinder] Размер контура: "
        << contour_.size() << '\n';

    const auto t0 = std::chrono::high_resolution_clock::now();

    // ── Первый проход: ищем минимальное De (snake mode) ──────────────────────
    {
        struct DeCandidate
        {
            double    D_ab = std::numeric_limits<double>::max();
            cv::Point ptA{}, ptB{};
            int i = -1, j = -1;
        };

        const int max_pct = static_cast<int>(thresholds_.snake_high_threshold * 100.0);
        const int min_pct = 2;
        std::map<int, DeCandidate> candidatesByPct;
        for (int pct = min_pct; pct <= max_pct; ++pct)
            candidatesByPct[pct] = {};

        const int n = static_cast<int>(contour_.size());
        const int contour_step = (n / 10000) + 1;
        //const int contour_step = 1;
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

                const auto skd = computeStemKeyDistances(i, j);

                //const size_t k = gdc_->getIndexByDistanceInRange(skd.G_k, i, j);
                const size_t k = gdc_->getIndexByDistance(skd.G_k);
                leaf_points_.ptS = contour_[k];

                if (!checkStemFlatness())
                    continue;

                //const size_t c = gdc_->getIndexByDistanceInRange(skd.G_c, i, k);
                //const size_t d = gdc_->getIndexByDistanceInRange(skd.G_d, k, j);
                const size_t c = gdc_->getIndexByDistance(skd.G_c);
                const size_t d = gdc_->getIndexByDistance(skd.G_d);
                leaf_points_.ptC = contour_[c];
                leaf_points_.ptD = contour_[d];

                distances_.D_cd = cv::norm(leaf_points_.ptC - leaf_points_.ptD);
                if (distances_.D_cd < thresholds_.distance_epsilon)
                    continue;

                if (!checkMinWidthStem())
                    continue;

                if (!checkBorderPixels())
                    continue;

                if (!checkTPoint())
                    continue;

                // De найден для данного i — сохраняем и переходим к следующему i
                if (distances_.D_ab < candidatesByPct[ratio_pct].D_ab)
                {
                    candidatesByPct[ratio_pct].D_ab = distances_.D_ab;
                    candidatesByPct[ratio_pct].ptA = leaf_points_.ptA;
                    candidatesByPct[ratio_pct].ptB = leaf_points_.ptB;
                    candidatesByPct[ratio_pct].i = i;
                    candidatesByPct[ratio_pct].j = j;
                    break;
                }
            }
        }

        bool deFound = false;
        double sum_de = 0.0;
        for (const auto& [pct, candidate] : candidatesByPct)
        {
            if (candidate.D_ab >= distances_.MAX_DE)
                continue;

            distances_.De = candidate.D_ab;
            leaf_points_.ptSnakeDeA = candidate.ptA;
            leaf_points_.ptSnakeDeB = candidate.ptB;

            //sum_de += candidate.D_ab;
            std::cout << "[LeafPetiolePointPairFinder] De = " << distances_.De
            << " (порог " << pct << "%)\n";

            deFound = true;
            break;
        }
        //distances_.De = sum_de / candidatesByPct.size();
        //deFound = true;
        //std::cout << "[LeafPetiolePointPairFinder] De = " << distances_.De << "\n";
            //<< " (порог " << pct << "%)\n";

        if (!deFound)
        {
            std::cout << "[LeafPetiolePointPairFinder] De не найдено ни для одного порога\n";
            return {};
        }
    }

    // ── Второй проход: ищем кандидата с максимальным T ───────────────────────
    Candidate cand1 = runScanPass();

    const auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "[LeafPetiolePointPairFinder] Время основного прохода: "
        << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
        << " мс\n";

    if (!cand1.valid)
        return {};

    // Сохраняем индексы лучшей первой пары
    indexes_.best_i_1 = cand1.idx_i;
    indexes_.best_j_1 = cand1.idx_j;

    std::cout << "i_1: " << indexes_.best_i_1 << std::endl;
    std::cout << "i_1: " << indexes_.best_j_1 << std::endl;
    // ── Ранний выход: T высокий — уверенный результат ─────────────────────────
    if (cand1.tp.T >= thresholds_.T_accept_threshold)
        return { cand1.ptA, cand1.ptB, cand1.ptS, true, cand1.tp };

    // ── Ранний выход: T слишком низкий ───────────────────────────────────────
    if (cand1.tp.T <= thresholds_.T_reject_threshold)
        return { cand1.ptA, cand1.ptB, cand1.ptS, false, cand1.tp };

    if (cand1.tp.T == 0.0)
        return { cand1.ptA, cand1.ptB, cand1.ptS, false, cand1.tp };
     
    // ── Третий проход: ищем альтернативного кандидата ────────────────────────
    Candidate cand2 = runScanPass(
        /*ref_ptS=*/&cand1.ptS,
        /*D_max=*/distances_.D_max);

    if (cand2.valid)
    {
        indexes_.best_i_2 = cand2.idx_i;
        indexes_.best_j_2 = cand2.idx_j;
    }

    std::cout << "i_2: " << indexes_.best_i_2 << std::endl;
    std::cout << "i_2: " << indexes_.best_j_2 << std::endl;

    //FindResult result;
    //result.ptA = cand1.ptA;
    //result.ptB = cand1.ptB;
    //result.ptS = cand1.ptS;
    //result.found = true;
    //result.tp1 = cand1.tp;
    //result.ptA2 = cand2.ptA;
    //result.ptB2 = cand2.ptB;
    //result.ptS2 = cand2.ptS;
    //result.tp2 = cand2.tp;
    //return result;

    // ── Принятие решения ──────────────────────────────────────────────────────
    if (!cand2.valid)
    {
        // Только один кандидат — спрашиваем SVM
        return { cand1.ptA, cand1.ptB, cand1.ptS, svmPredict(cand1.tp), cand1.tp };
    }

    const double dT = std::abs(cand1.tp.T - cand2.tp.T);

    if (dT >= thresholds_.T_epsilon)
    {
        // Кандидаты сильно различаются по T — спрашиваем SVM про кандидата с наибольшим T
        return { cand1.ptA, cand1.ptB, cand1.ptS, svmPredict(cand1.tp), cand1.tp, cand2.tp, cand2.ptA, cand2.ptB, cand2.ptS };
    }

    // Кандидаты близки по T — используем SVM для обоих
    const bool p1 = svmPredict(cand1.tp);
    const bool p2 = svmPredict(cand2.tp);

    if (p1 && !p2)
        return { cand1.ptA, cand1.ptB, cand1.ptS, true, cand1.tp, cand2.tp, cand2.ptA, cand2.ptB, cand2.ptS };
    if (!p1 && p2)
        return { cand2.ptA, cand2.ptB, cand2.ptS, true, cand2.tp, cand1.tp, cand1.ptA, cand1.ptB, cand1.ptS };
    if (!p1 && !p2)
        return {}; // Оба отклонены

    // Оба приняты SVM — выбираем по decision function
    const float d1 = svmDecision(cand1.tp);
    const float d2 = svmDecision(cand2.tp);
    const Candidate& winner = (std::abs(d1) >= std::abs(d2)) ? cand1 : cand2;
    return { winner.ptA, winner.ptB, winner.ptS, true };
}