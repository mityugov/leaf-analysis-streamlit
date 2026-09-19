#include "./Utils.h"


bool isTwoLinesIntersection(cv::Point2f l1, cv::Point2f l2, cv::Point2f l3, cv::Point2f l4) {
	cv::Point2f l12dif = l1 - l2;
	cv::Point2f l34dif = l3 - l4;

	double d = l12dif.x * l34dif.y - l12dif.y * l34dif.x;
	if (abs(d) <= std::numeric_limits<double>::min()) {
		return false;
	}
	double first = (l1.x * l2.y - l1.y * l2.x);
	double second = (l3.x * l4.y - l3.y * l4.x);
	double x = (first * l34dif.x - second * l12dif.x) / d;
	double y = (first * l34dif.y - second * l12dif.y) / d;

	bool insideFirst = isInInterval(x, l1.x, l2.x) && isInInterval(y, l1.y, l2.y);
	bool insideSecond = isInInterval(x, l3.x, l4.x) && isInInterval(y, l3.y, l4.y);
	return insideFirst && insideSecond;
}


//TODO: переделать данный метод: Алгоритм должен работать с вещественными числами.
// Метод возвращает список направлений из center, формирующих окружность.
// Основа DDA-алгоритма.
std::vector<cv::Point2f> getCircleDirections(double radius) {
	std::vector<cv::Point2f> result;
	double x = 0;
	double y = radius;
	double delta = 1 - 2 * radius;
	double error = 0;
	while (y >= x) {
		result.emplace_back(+x, +y);
		result.emplace_back(+x, -y);
		result.emplace_back(-x, +y);
		result.emplace_back(-x, -y);
		result.emplace_back(+y, +x);
		result.emplace_back(+y, -x);
		result.emplace_back(-y, +x);
		result.emplace_back(-y, -x);
		error = 2 * (delta + y) - 1;
		if ((delta < 0) && (error <= 0)) {
			delta += 2 * ++x + 1;
			continue;
		}
		if ((delta > 0) && (error > 0)) {
			delta -= 2 * --y + 1;
			continue;
		}
		delta += 2 * (++x - --y);
	}
	return result;
}


std::vector<cv::Point2f> getCorrectCircleDirections(
	double radius,
	double alpha,
	cv::Point2f defaultDirection
) {
	double limitCosinus = cos(alpha * PI / 180.0);
	std::vector<cv::Point2f> result;
	for (cv::Point2f direction : getCircleDirections(radius)) {
		if (cos_about_two_vectors(direction, defaultDirection) >= limitCosinus) {
			result.push_back(direction);
		}
	}
	return result;
}

cv::Point2f ddaBase(cv::Point2f center, const cv::Mat& img, cv::Point2f direction) {
	// TODO: оптимизировать round (это очень тормозит программу). 
	float dX = direction.x;
	float dY = direction.y;
	float L = abs(dY) >= abs(dX) ? abs(dY) : abs(dX);
	dY /= L;
	dX /= L;
	float x = center.x;
	float y = center.y;
	cv::Point p;
	do
	{
		x += dX;
		y += dY;
		p.x = round(x);
		p.y = round(y);
	} while (isPointInside(p, img));
	return cv::Point2f(round(x - dX), round(y - dY));
}

std::tuple<cv::Point2f, cv::Point2f, float> dda_line(cv::Point2f center, const cv::Mat& img, cv::Point2f direction) {
	//	dY = -direction.x;
	//  dX = direction.y;
	cv::Point2f correctedDirection = cv::Point2f(direction.y, -direction.x);
	cv::Point2f first = ddaBase(center, img, correctedDirection);
	cv::Point2f second = ddaBase(center, img, -1 * correctedDirection);
	float f_dist = euclid_dst(first, center);
	float s_dist = euclid_dst(second, center);
	return std::tuple(first, second, std::min(f_dist, s_dist) / std::max(f_dist, s_dist));
}

std::pair<int, int> getIndexesByDirection(
	cv::Point2f direction,
	cv::Point2f center,
	const cv::Mat& filledExternalContourImg,
	const ContourIndexer& indexer
) {
	direction = cv::Point2f(direction.y, -direction.x);
	cv::Point2f first = ddaBase(center, filledExternalContourImg, direction);
	cv::Point2f second = ddaBase(center, filledExternalContourImg, -1 * direction);
	return { indexer.findContourPoint(first.x, first.y), indexer.findContourPoint(second.x, second.y) };
}