#include "MultiContourController.h"
#include "Utils.h"

using SimpleLine = std::pair<cv::Point, cv::Point>;
using ImageContour = std::vector<cv::Point>;

std::vector<cv::Point> findPointsFromSector(
	int high,
	int low,
	std::pair<int, int> markers,
	const std::vector<cv::Point>& contour
) {
	std::vector<cv::Point> result;

	if (low >= high) {
		std::swap(low, high);
	}

	if (!(low <= markers.first && markers.first <= high && low <= markers.second && markers.second <= high)) {
		for (size_t j = low; j <= high; j++) {
			result.push_back(contour[j]);
		}
	}
	else {
		//m1 или m2 в пределах [low, high], тогда нужно от [0, low] и от [high, size]
		for (size_t i = 0; i < contour.size(); i++) {
			if (i <= low || i >= high) {
				result.push_back(contour[i]);
			}
		}
	}
	return result;
}

std::vector<cv::Point> getPointsBeetwen(std::pair<int, int> currentLine,
	std::pair<int, int> previousLine,
	const std::vector<cv::Point>& contour
) {
	std::vector<cv::Point> result = findPointsFromSector(currentLine.first, previousLine.first, { currentLine.second, previousLine.second }, contour);
	std::vector<cv::Point> second = findPointsFromSector(currentLine.second, previousLine.second, { currentLine.first, previousLine.first }, contour);
	result.insert(result.end(), second.begin(), second.end());
	return result;
}

MultiContourImageRectifier::MultiContourImageRectifier(const cv::Mat& img) {
	std::vector<std::vector<cv::Point>> allContours;
	cv::findContours(img, allContours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE, cv::Point(0, 0));
	imgWithContours = cv::Mat::zeros(img.size(), CV_8UC1); 

	externalContour = *std::max_element(allContours.begin(), allContours.end(), []
	(const std::vector<cv::Point>& first, const std::vector<cv::Point>& second) { return first.size() < second.size(); });

	for (size_t i = 0; i < allContours.size(); i++) {
		auto contour = allContours[i];
		RectifyContourContainer* container = new RectifyContourContainer(contour);
		contours.push_back(container);
		for (auto p : contour) {
			imgWithContours.at<uchar>(p) = 255; 
			if (pointToContour.count(p) == 0) {
				pointToContour.insert({ p, {container} });
			}
			else {
				pointToContour.at(p).push_back(container);
			}
		}
	}
}

MultiContourImageRectifier::~MultiContourImageRectifier() {
	for (size_t i = 0; i < contours.size(); i++) {
		delete contours[i]; 
	}
}

inline cv::Point rotatePoint(cv::Point point, double angle) {
	cv::Point2f direction = point; 
	double cosinus = cos(angle); 
	double sinus = sin(angle); 
	return cv::Point(round(direction.x * cosinus - direction.y * sinus),
					 round(direction.y * cosinus + direction.x * sinus));
}


//TODO: подумать, как ускорить. Очень тяжелый метод
cv::Mat getMask(
	const std::vector<cv::Point>& detectedPoints,
	const std::vector<SimpleLine>& borderedLines,
	const cv::Size& maskSize
) {
	cv::Mat mask = cv::Mat::zeros(maskSize, CV_8UC1);
	std::for_each(detectedPoints.begin(), detectedPoints.end(), [&mask](const cv::Point& p) { mask.at<uchar>(p) = 255; });
	std::for_each(borderedLines.begin(), borderedLines.end(), 
		[&mask](const SimpleLine& line)
		{ 
			cv::line(mask, line.first, line.second, 255); 
		}
	);
	std::vector<cv::Vec4i> hierarchy;
	std::vector<std::vector<cv::Point>> allContours;
	cv::findContours(mask, allContours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point(0, 0));
	cv::drawContours(mask, allContours, 0, 255, cv::FILLED);

	return mask;
}




void MultiContourImageRectifier::rectifySector(
	std::pair<int, int> currentLine,
	std::pair<int, int> previousLine,
	double angle, cv::Point rotateCenter, cv::Point newCenter) {
	auto pointsFromSectors = getPointsBeetwen(currentLine, previousLine, externalContour); 
	if (pointsFromSectors.empty()) {
		return; 
	}


	std::vector<SimpleLine> lines; 
	lines.push_back({ externalContour[currentLine.first], externalContour[currentLine.second] }); 
    lines.push_back({ externalContour[previousLine.first], externalContour[previousLine.second] });
	cv::Mat mask;
	cv::bitwise_and(getMask(pointsFromSectors, lines, imgWithContours.size()), imgWithContours, mask); 

	std::vector<cv::Point> pointsFromMask;
	cv::findNonZero(mask, pointsFromMask); 

	for (const cv::Point& point : pointsFromMask) {
		if (pointToContour.count(point) == 0) {
			continue; 
		}
		for (auto contour : pointToContour.at(point)) {
			rectifyPoint(contour, point, angle, rotateCenter, newCenter);
		}
	}
}


void MultiContourImageRectifier::rectifyTerminalSector(
	std::pair<int, int> terminalLine, 
	std::pair<int, int> previousLine,
	double angle, cv::Point rotateCenter, cv::Point newCenter) {

	auto pointsFromSectors = findPointsFromSector(terminalLine.first, terminalLine.second, previousLine, externalContour);
	if (pointsFromSectors.empty()) {
		return;
	}

	std::vector<SimpleLine> lines;
	lines.push_back({ externalContour[terminalLine.first], externalContour[terminalLine.second] });
	cv::Mat mask;
	cv::bitwise_and(getMask(pointsFromSectors, lines, imgWithContours.size()), imgWithContours, mask);

	std::vector<cv::Point> pointsFromMask;
	cv::findNonZero(mask, pointsFromMask);

	for (const cv::Point& point : pointsFromMask) {
		if (pointToContour.count(point) == 0) {
			continue;
		}
		for (auto contour : pointToContour.at(point)) {
			rectifyPoint(contour, point, angle, rotateCenter, newCenter);
		}
	}
}

void MultiContourImageRectifier::rectifyOther(double angle, cv::Point rotateCenter, cv::Point newCenter) {
	for (auto contour : contours) {
		for (size_t i = 0; i < contour->size(); i++) {
			rectifyPoint(contour, (*contour)[i], angle, rotateCenter, newCenter); 
		}
	}
}

void MultiContourImageRectifier::rectifyPoint(RectifyContourContainer* contour, cv::Point point, double angle, cv::Point rotateCenter, cv::Point newCenter) {
	int idx = contour->rectifyPoint(point, angle, rotateCenter, newCenter);
	if (idx != -1) {
		auto rectifiedPoint = contour->getRectified(idx);
		min_x = std::min(min_x, rectifiedPoint.x);
		min_y = std::min(min_y, rectifiedPoint.y);
		max_x = std::max(max_x, rectifiedPoint.x);
		max_y = std::max(max_y, rectifiedPoint.y);
	}
}



void traverseContours(const std::vector<RectifyContourContainer*>& contours, const std::vector<cv::Vec4i>& hierarchy, int idx, int level,
	                  cv::Mat& img) {
	if (idx == -1) return; // нет контура

	RectifyContourContainer* container = contours[idx];
	container->draw(img, (level % 2 == 0) ? cv::Scalar(255, 255, 255) : cv::Scalar(0, 0, 0));


	// Рекурсивно обходим дочерние контуры (первый дочерний элемент)
	traverseContours(contours, hierarchy, hierarchy[idx][2], level + 1, img);

	// Обходим следующий контур на том же уровне
	traverseContours(contours, hierarchy, hierarchy[idx][0], level, img);
}


cv::Mat MultiContourImageRectifier::drawRectified() {
	int rows = max_y - min_y + 1;
	int cols = max_x - min_x + 1;

	for (auto contour : contours) {
		contour->resize(min_y, min_x);
	}

	for (auto contour : contours) {
		contour->applyOffset(min_y, min_x);
	}

	cv::Mat result = cv::Mat::zeros(rows, cols, CV_8UC1);
	for (size_t i = 0; i < hierarchy.size(); ++i) {
		if (hierarchy[i][3] == -1) {
			traverseContours(contours, hierarchy, i, 0, result);
		}
	}

	return result; 
}



RectifyContourContainer::RectifyContourContainer(std::vector<cv::Point> contour) : contour(contour) {
	rectifyMask.resize(contour.size(), 0);
	rectifiedContour.resize(contour.size(), cv::Point(0, 0));
}


// Возвращает набор точек от p1, до p2 (не включая p1, p2). 
std::vector<cv::Point> detectPoints(cv::Point p1, cv::Point p2) {
	cv::Point2f direction = p2 - p1; 
	float dX = direction.x;
	float dY = direction.y;
	float L = abs(dY) >= abs(dX) ? abs(dY) : abs(dX);
	dY /= L;
	dX /= L;
	float x = p1.x;
	float y = p1.y;
	cv::Point current = p1; 
	std::vector<cv::Point> result;
	do
	{
		x += dX;
		y += dY;
		current.x = round(x);
		current.y = round(y);
		if (current != p2) {
			result.emplace_back(current);
		}

	} while (current != p2);
	return result;
}


void RectifyContourContainer::resize(int& min_y, int& min_x) {
	std::vector<cv::Point> normalizedContour; 
	cv::Point first = rectifiedContour[0]; 
	normalizedContour.push_back(first); 
	for (size_t i = 0; i < rectifiedContour.size(); i++) {
		cv::Point current = rectifiedContour[i]; 
		cv::Point next = (i == (rectifiedContour.size() - 1)) ? first : rectifiedContour[i + 1]; 

		if (euclid_dst(current, next) > 2) {
			for (cv::Point spacePoint : detectPoints(current, next)) {
				min_x = std::min(min_x, spacePoint.x);
				min_y = std::min(min_y, spacePoint.y); 
				normalizedContour.push_back(spacePoint);
			}
		}

		if (next != first) {
			normalizedContour.push_back(next);
		}
	}

	rectifiedContour = normalizedContour; 
}


void RectifyContourContainer::draw(cv::Mat& img, cv::Scalar color) {
	std::vector<std::vector<cv::Point>> contours = { rectifiedContour }; 
	cv::drawContours(img, contours, 0, color, cv::FILLED); 
}

int RectifyContourContainer::rectifyPoint(cv::Point point, double angle, cv::Point oldCenter, cv::Point newCenter) {
	cv::Point currentPoint;
	for (size_t i = 0; i < contour.size(); i++) {
		currentPoint = contour[i];
		if (currentPoint == point) {
			if (!rectifyMask[i]) {
				rectifyMask[i] = 1;
				rectifiedContour[i] = newCenter + rotatePoint(point - oldCenter, angle);
				return i;
			}
		}
	}
	return -1; 
}
