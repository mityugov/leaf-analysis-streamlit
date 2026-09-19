#include "CurvilinearSymmetryBase.h"
#include "./MultiContourController.h"
#include "Utils.h"
#include <numeric>
#include <functional>
#include "./Smoother.h"

//#define DEBUG
//#define DRAW


cv::Point2f find_condition(cv::Point2f center, cv::Mat& img) {
	float x_max = 0;
	float y_max = 0;
	float max = 0;
	float length; 
	LoopParameters<double> loopParameters = { -1, 1, 0.25 };

	for (float y = loopParameters.start; y <= loopParameters.end; y += loopParameters.step)
		for (float x = loopParameters.start; x <= loopParameters.end; x += loopParameters.step) {
			if (isAbsoluteSmallOf(x, loopParameters.step) && isAbsoluteSmallOf(y, loopParameters.step)
				|| !isPointInside(cv::Point2f(x + center.x, y + center.y), img)) {
				continue; 
			}
			length = euclid_dst(center, ddaBase(center, img, cv::Point2f(x, y)));
			if (length > max) {
				max = length;
				x_max = x;
				y_max = y;
			}
		}
	return cv::Point2f(x_max, y_max);
}


// Возвращает локальную меру Жаккара или -1, если direction не проходит проверку.
// Корректируется direction на основе previousDirection (становится normalizedDirection).
double getLocalJaccardFor(
	const cv::Point2f center, 
	const cv::Point2f correctedDirection, 
	const cv::Mat& img,
	double& lineSize // Возврат к предыдущему алгоритму 
) {
	lineSize = 0;
	if (length(correctedDirection) < 1) {
		return -1;
	}
	cv::Point2f newCenter = correctedDirection + center;
	if (!isPointInside(newCenter, img)) {
		return -1; 
	}

	if (distanceFromPointByDirection(center, correctedDirection, img) < euclid_dst(newCenter, center)) {
		return -1;
	}

	double currentLocalJaccard; 
	cv::Point2f f, s;
	std::tie(f, s, currentLocalJaccard) = dda_line(newCenter, img, correctedDirection);
	lineSize = euclid_dst(f, s); 
	return currentLocalJaccard; 
}

std::vector<double> localJaccard;

std::tuple<cv::Point2f, std::pair<int, int>> GreedySearch::findNextDirection(cv::Point2f center,
	cv::Point2f previousDirection,
	cv::Mat img, double h, float alpha_degrees, double lambda, 
	bool isFirstCenter, std::pair<int, int> prevStepLine,
	const std::vector<std::pair<cv::Point2f, cv::Point2f>> previousLines)
{
	cv::Point2f normalizedDirection, correctedDirection, bestDirection(0, 0), prevDirection;
	double currentLocalJaccard, currentMark, bestJaccard = 0, bestMark = 0;
	std::pair<int, int> currentLine, prevLine, bestLine; 

	int parts; 	
	double smoothing; 
	double currentLength, maxLength;
	auto directions = getCorrectCircleDirections(h, alpha_degrees, previousDirection); 
	std::vector<double> directionLengthes; 
	double localMaxLength;
	double maxDirectionLength = 0; 
	for (const auto& direction : directions) {
		localMaxLength = distanceFromPointByDirection(center, direction, img); 
		maxDirectionLength = std::max(maxDirectionLength, localMaxLength);
		directionLengthes.push_back(localMaxLength);
	}

	double lengthCorrection;
	cv::Point2f newDirection; 
	for (size_t i = 0; i < directions.size(); i++) {
		newDirection = directions[i];
		localMaxLength = directionLengthes[i]; 

		lengthCorrection = localMaxLength / maxDirectionLength;
		maxLength = std::max(localMaxLength, h); 
		parts = int(round(maxLength / h)); 
		double length; 
		for (int partIdx = 1; partIdx < parts; partIdx++) {
			currentLength = h * partIdx; 
			prevDirection = resize(previousDirection, currentLength); 
 			normalizedDirection = resize(newDirection, currentLength);
			correctedDirection = resize(((1.0 - lambda) * normalizedDirection + lambda * prevDirection), currentLength);
			currentLocalJaccard = getLocalJaccardFor(center, correctedDirection, img, length);
			if (currentLocalJaccard < 0) {
				continue;
			}

			currentLine = getIndexesByDirection(correctedDirection, center + correctedDirection, imgWithOnlyContour, *externalContourIndexer); 
			if (!isFirstCenter && isAnyIntersection(previousLines, { contour[currentLine.first], contour[currentLine.second] })) {
				continue; 
			}
	
			currentMark = currentLocalJaccard * (1.0 - double(partIdx) / double(parts));
			if (currentMark > bestMark) {
				bestMark = currentMark;
				bestDirection = correctedDirection;
				bestLine = currentLine; 
				bestJaccard = currentLocalJaccard;
			}
		}
	}

	localJaccard.push_back(bestJaccard);
	return std::tuple(bestDirection, bestLine);
}


std::vector<cv::Point2f> GreedySearch::algorithm(Data& dt, cv::Point2f start_condition)
{
	localJaccard.clear();
	cv::Point2f current_point = dt.start_point_;
	float alpha;
	cv::Point2f last_condition;
	cv::Point2f current_condition = start_condition;
	std::vector<cv::Point2f> result;
	std::vector<std::pair<cv::Point2f, cv::Point2f>> points; 
	std::pair<int, int> previousIndexes; 
	std::pair<int, int> currentIndexes(-1, -1);
	float h;
	bool isFirstCenter;
	do
	{
		last_condition = current_condition;
		previousIndexes = currentIndexes; 
		result.push_back(current_point);
		h = sqrt(distanceFromPointByDirection(current_point, current_condition, dt.image_));
		alpha = round(90.0 / std::max(log(h), float(2.0))); 
		isFirstCenter = result.size() == 1; 
		if (!isFirstCenter) {
			points.push_back({ externalContourIndexer->get(previousIndexes.first), externalContourIndexer->get(previousIndexes.second) });
		}
		std::tie(current_condition, currentIndexes) = findNextDirection(
			current_point, 
			current_condition, 
			dt.image_, 
			h, 
			alpha, 
			dt.lambda_, 
			isFirstCenter, 
			previousIndexes, 
			points
		);
		/*
		if (false && euclid_dst(current_point, current_point + current_condition) > h && isPointInside(current_point + current_condition, dt.image_)) {
			if (isFirstCenter) {
				previousIndexes = getIndexesByDirection(current_condition, current_point, imgWithOnlyContour, *externalContourIndexer);
			}

			Smoother smoother = Smoother(dt.image_, imgWithOnlyContour, externalContourIndexer, points);
			cv::Point2f nextPoint = current_point + current_condition;
			auto atoms = smoother.smooth(h, currentIndexes, previousIndexes, current_point, nextPoint, last_condition);
			for (const auto& atom : atoms) {
				result.push_back(atom.point);
				points.push_back(atom.contourPoints);
				current_point = atom.point; 
				current_condition = atom.directionToPoint;
			}
		}
		*/

		current_point += current_condition;
	} while (length(current_condition) >= 1 && isPointInside(current_point, dt.image_));
	return result;
}


void CurvilinearSymmetryBase::Draw(Data& dt, Curvilinear& line)
{
	cv::Mat explain = cv::Mat::zeros(dt.image_.size(), CV_8UC3);
	explain.setTo(cv::Scalar(255, 255, 255), dt.image_ == 0);
	for (size_t i = 1; i < line.curvilinearAxisAtoms.size(); i++)
		cv::line(explain, line.curvilinearAxisAtoms[i - 1], line.curvilinearAxisAtoms[i], cv::Scalar(0, 255, 255), 2, 4);

	cv::Scalar color = cv::Scalar(0, 0, 255); 
	for (size_t i = 0; i < line.start_points_.size(); i++) {
		cv::circle(explain, line.start_points_[i], 3, color, cv::FILLED);
	}
	cv::circle(explain, line.curvilinearAxisAtoms[0], 3, cv::Scalar(255, 0, 0), cv::FILLED);
	line.curvilinear_axis_ = explain; 

	//cv::imshow("Axis", explain); 
	//cv::waitKey(0); 
	//cv::destroyAllWindows(); 


	line.straightened_ = cv::Mat::zeros(line.rectified_image_.size(), CV_8UC3);
	line.straightened_.setTo(cv::Scalar(255, 255, 255), line.rectified_image_ == 0);
	cv::line(line.straightened_, line.line_of_rectified_.getFirstPoint(), line.line_of_rectified_.getSecondPoint(), cv::Scalar(0, 255, 255));

	//TODO: вот тут надо переделать, так как сейчас не всё отраженное изображение попадает в "кадр". 
	cv::Mat img1;
	cv::flip(line.rectified_image_, img1, 1);
	Line axis = line.line_of_rectified_;
	int difference = line.rectified_image_.cols - 1 - 2 * axis.getFirstPoint().x;
	line.reflected_ = cv::Mat(line.rectified_image_.size(), CV_8UC3, cv::Scalar(255, 255, 255));
	line.reflected_.setTo(cv::Scalar(255, 0, 0), line.rectified_image_ != 0);

	for (int i = 0; i < line.rectified_image_.rows; i++)
	{
		for (int j = 0; j < line.rectified_image_.cols; j++)
		{
			cv::Point offset_current = cv::Point(j - difference, i);
			if (offset_current.x < 0 || offset_current.x >= line.rectified_image_.cols || img1.at<uchar>(i, j) == 0)
				continue;
			line.reflected_.at<cv::Vec3b>(offset_current) = line.rectified_image_.at<uchar>(offset_current) == 0 ? cv::Vec3b(0, 0, 255) 
																												 : cv::Vec3b(0, 0, 0);
		}
	}
	cv::line(line.reflected_, axis.getFirstPoint(), axis.getSecondPoint(), cv::Scalar(0, 255, 255));


	/*
	//TODO: новый алгоритм получения reflected-изображения с использованием forEach на изображении.
	int axisX = axis.getFirstPoint().x; 
	int newCols = 2 * std::max(axisX, line.rectified_image_.cols - axisX);
	cv::Mat newReflected = cv::Mat(line.rectified_image_.rows, newCols, CV_8UC3, cv::Scalar(255, 255, 255)); 
	
	int offsetX = newCols / 2 - std::min(axisX, line.rectified_image_.cols - axisX);
	uchar b, r; 
	for (int i = 0; i < line.rectified_image_.rows; i++)
	{
		for (int j = 0; j < line.rectified_image_.cols; j++)
		{			
			b = line.rectified_image_.at<uchar>(i, j); 
			r = img1.at<uchar>(i, j + offsetX);
			if (b && r) {
				newReflected.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0); 
			} else if (b) {
				newReflected.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 0, 0);
			} else if (r) {
				newReflected.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 255);
			}
		}
	}
	cv::imwrite("test.bmp", newReflected);
	*/
}




double getAngleForVector(cv::Point2f normal) {
	if (normal.y != 0.0 && normal.x != 0.0) {
		return normal.x > 0.0 ? (normal.y > 0.0 ? atan(normal.x / normal.y) + PI : atan(-normal.y / normal.x) + PI * 3.0 / 2.0)
							  : (normal.y < 0.0 ? atan(normal.x / normal.y) : atan(normal.y / -normal.x) + PI / 2.0);
	}
	return normal.x != 0.0 ? (normal.x < 0.0 ? (PI / 2.0) : (PI * 3.0 / 2.0)) :
							 (normal.y > 0.0 ? PI : 0.0); 
}

using PointTranslator = std::function<cv::Point(cv::Point)>; 


PointTranslator getBaseTranslator(cv::Point oldCenter, cv::Point newCenter, float angle) {
	return [oldCenter, newCenter, angle](cv::Point contourPoint) {
		auto p = contourPoint - oldCenter;
		return cv::Point(newCenter.x + round(p.x * cos(angle) - p.y * sin(angle)),
					  newCenter.y + round(p.y * cos(angle) + p.x * sin(angle)));
	};
}

class ExternalContourImageRectifier {
	std::vector<cv::Point> contour; 
	std::vector<cv::Point> rectifiedContour; 
	std::vector<int> contourMask; 
	PointTranslator translator; 

	int min_x; 
	int min_y;
	int max_x;
	int max_y; 

	void rectify(
		int highBound,
		int lowBound
	) {
		cv::Point p;
		for (int j = lowBound; j <= highBound; j++)
		{
			if (!contourMask[j]) {
				p = translator(contour[j]); 
				contourMask[j] = 1;
				rectifiedContour[j] = p;
				min_x = std::min(min_x, p.x); 
				min_y = std::min(min_y, p.y); 
				max_x = std::max(max_x, p.x);
				max_y = std::max(max_y, p.y);
			}
		}
	}

	void rectifySector(int high, int low, std::pair<int, int> markers) {
		if (low >= high) {
			std::swap(low, high);
		}

		if (!(isInInterval(markers.first, low, high) && isInInterval(markers.second, low, high))) {
			rectify(high, low);
		}
		else {
			//m1 или m2 в пределах [low, high], тогда нужно от [0, low] и от [high, size]
			rectify(low, 0);
			rectify(contour.size() - 1, high);
		}
	}

public:
	ExternalContourImageRectifier(std::vector<cv::Point> contour) : 
		contour(contour)
	{
		min_x = max_x = contour[0].x;
		min_y = max_y = contour[0].y;
		contourMask.resize(contour.size(), 0);
		rectifiedContour.resize(contour.size(), cv::Point(0, 0));
	}

	void rectifyOther(PointTranslator translator) {
		this->translator = translator; 
		rectify(contour.size() - 1, 0);
		for (auto& p : rectifiedContour) {
			p.x -= min_x;
			p.y -= min_y;
		}
	}

	cv::Point2f getOffset() {
		return cv::Point2f(min_x, min_y); 
	}

	cv::Mat getRectifiedImage() {
		int rows = max_y - min_y + 1;
		int cols = max_x - min_x + 1;
		cv::Mat img = cv::Mat::zeros(rows, cols, CV_8UC1);
		cv::Point p, p_next;
		for (size_t i = 0; i < rectifiedContour.size(); i++)
		{
			p = rectifiedContour[i];
			p_next = (i == (rectifiedContour.size() - 1)) ? rectifiedContour[0] : rectifiedContour[i + 1];
			cv::line(img, p, p_next, 255);
		}
		std::vector<std::vector<cv::Point>> contours;
		contours.push_back(Data::findExternalContour(img));
		cv::drawContours(img, contours, 0, 255, cv::FILLED);
		return img;
	}


	void rectifySector(
		std::pair<int, int> indexesPair,
		std::pair<int, int> prevIndexes,
		PointTranslator translator
	) {
		this->translator = translator;
		rectifySector(indexesPair.first, prevIndexes.first, { indexesPair.second, prevIndexes.second });
		rectifySector(indexesPair.second, prevIndexes.second, { indexesPair.first, prevIndexes.first });
	}

	void rectifyTerminalSector(
		std::pair<int, int> terminalLine,
		std::pair<int, int> previousLine,
		PointTranslator translator
	) {
		this->translator = translator; 
		rectifySector(terminalLine.first, terminalLine.second, previousLine);
	}

};

cv::Point2f findIntersectionWithNormal(const cv::Point2f& firstLinePoint,
	const cv::Point2f& secondLinePoint,
	const cv::Point2f& C) {

	cv::Point2f u = secondLinePoint - firstLinePoint;
	cv::Point2f n(-u.y, u.x);

	// Вектор от A к C
	cv::Point2f v = C - firstLinePoint;

	// Альтернативный способ через параметр t на прямой AB
	float t = (v.x * u.x + v.y * u.y) / (u.x * u.x + u.y * u.y);
	return firstLinePoint + u * t;
}

bool isPointInsideSection(cv::Point2f point, cv::Point2f a, cv::Point2f b) {
	cv::Point2f u = b - a;
	cv::Point2f AN = point - a;
	float t = (AN.x * u.x + AN.y * u.y) / (u.x * u.x + u.y * u.y);
	return t >= 0.0f && t <= 1.0f;
}


class BordersController {
	using SimpleLine = std::pair<cv::Point, cv::Point>;
	using Borders = std::pair<SimpleLine, SimpleLine>;
	std::vector<Borders> borders;


private:
	ContourIndexer* indexer; 

	SimpleLine apply(std::pair<int, int> indexes, PointTranslator translator) {
		cv::Point2f first = translator(indexer->get(indexes.first));
		cv::Point2f second = translator(indexer->get(indexes.second)); 
		return SimpleLine(first, second); 
	}


public:
	BordersController(ContourIndexer* indexer) : indexer(indexer) {}

	void addTerminator(std::pair<int, int> indexes, PointTranslator translator) {
		auto line = apply(indexes, translator);
		borders.push_back(Borders(line, line)); 
	}

	void add(std::pair<int, int> firstBound, std::pair<int, int> secondBound, PointTranslator translator) {
		auto first = apply(firstBound, translator); 
		auto second = apply(secondBound, translator);
		borders.push_back(Borders(first, second)); 
	}

	void applyOffset(cv::Point offset) {
		for (size_t i = 0; i < borders.size(); i++) {
			const auto& border = borders[i]; 
			if (i == 0 || i == (borders.size() - 1)) {
				auto line = border.first; 
				SimpleLine newLine = SimpleLine(line.first - offset, line.second - offset); 
				borders[i] = Borders(newLine, newLine); 
			}
			else {
				auto firstLine = border.first; 
				SimpleLine newFirstLine = SimpleLine(firstLine.first - offset, firstLine.second - offset);

				auto secondLine = border.second; 
				SimpleLine newSecondLine = SimpleLine(secondLine.first - offset, secondLine.second - offset);

				borders[i] = Borders(newFirstLine, newSecondLine); 
			}
		}
	}

	cv::Mat getImgWithBorders(cv::Mat img) {
		cv::Mat resultImg = cv::Mat::zeros(img.size(), CV_8UC3); 
		resultImg.setTo(cv::Scalar(255, 255, 255), img != 0);
		for (size_t i = 0; i < borders.size(); i++) {
			const auto& border = borders[i];
			auto line = border.first;
			cv::line(resultImg, line.first, line.second, cv::Scalar(0, 0, 255));

			if (i != 0 && (i != (borders.size() - 1))) {
				line = border.second;
				cv::line(resultImg, line.first, line.second, cv::Scalar(255, 0, 0));
			}
		}
		return resultImg;
	}
 };



PointTranslator getNewTranslator(double lengthFromStart, cv::Point2f firstSectorPoint, cv::Point2f secondSectorPoint) {
	return [lengthFromStart, firstSectorPoint, secondSectorPoint](cv::Point contourPoint) {
		cv::Point2f sectorProjection = findIntersectionWithNormal(firstSectorPoint, secondSectorPoint, contourPoint); 

		if (isPointInsideSection(sectorProjection, firstSectorPoint, secondSectorPoint)) {
			cv::Point result;
			result.x = -round(calculateDistanceFromPointToLine(contourPoint, firstSectorPoint, secondSectorPoint));
			result.y = round(lengthFromStart + euclid_dst(sectorProjection, firstSectorPoint));
			return result;
		}
		else {
			cv::Point2f end;
			if (euclid_dst(sectorProjection, firstSectorPoint) < euclid_dst(sectorProjection, secondSectorPoint)) {
				end = firstSectorPoint;
			}
			else {
				end = secondSectorPoint;
			}





			// Отдельная обработка для точек, которые не имеют нормали в отрезок (first, second);
			// TODO: пока что будем считать просто второй точкой second для отладки. 
			return cv::Point(0, round(lengthFromStart + euclid_dst(firstSectorPoint, end)));
		}
	};
}

cv::Mat CurvilinearSymmetryBase::Cut(const cv::Mat& img,
	const std::vector<cv::Point2f>& centers,
	const std::vector<cv::Point>& contour,
	Line& line_of_rectified_)
{
	//MultiContourImageRectifier controller = MultiContourImageRectifier(img);
	ExternalContourImageRectifier controller = ExternalContourImageRectifier(contour);

	cv::Point2f next_center = centers[0];
	float angle = 0;
	cv::Point2f normal;
	cv::Point2f center;
	cv::Point2f prevCenter; 
	std::pair<int, int> prevIndexes, indexesPair;

	for (size_t i = 0; i < centers.size(); i++) {
		center = centers[i];
		normal = (i == 0) ? (centers[i + 1] - center) : (center - prevCenter);
		indexesPair = getIndexesByDirection(normal, center, imgWithOnlyContour, *externalContourIndexer);
		angle = getAngleForVector(normal);

		if (i == 0) {
			prevCenter = centers[++i]; 
			prevIndexes = getIndexesByDirection(normal, prevCenter, imgWithOnlyContour, *externalContourIndexer);
			auto translator = getBaseTranslator(next_center, next_center, angle); 
			controller.rectifyTerminalSector(indexesPair, prevIndexes, translator);
			std::swap(prevIndexes, indexesPair); 
			std::swap(prevCenter, center); 
		}

		auto translator = getBaseTranslator(prevCenter, next_center, angle); 
		controller.rectifySector(indexesPair, prevIndexes, translator);
		next_center.y -= euclid_dst(center, prevCenter);
		prevIndexes = indexesPair; 
		prevCenter = center;
	}

	auto translator = getBaseTranslator(prevCenter, next_center, angle); 
	controller.rectifyOther(translator);
	auto offset = controller.getOffset(); 
	line_of_rectified_.setLine(
		centers[0] - offset, 
		next_center - offset
	);
	auto result = controller.getRectifiedImage(); 
	return result;
}



void CurvilinearSymmetryBase::writeResult(Data& dt, Curvilinear& line) {
	Draw(dt, line); 
	std::cout << "Lambda is " << dt.lambda_ << std::endl;
	std::cout << "Step length = "; 
	std::cout << ((dt.stepLength > 0) ? std::to_string(dt.stepLength) : "auto") << std::endl;
	if (!dt.definedInitialPoint) {
		std::cout << "Initial point = " << line.curvilinearAxisAtoms[0] + cv::Point2f(dt.offset_) << std::endl; 
	}
	std::cout << line << std::endl;
}

std::vector<point_info> UserInputGenerator::getInitialPoints(Data& dt) {
	std::vector<point_info> result;
	point_info current(dt.start_point_); 
	current.setMaxDirection(0, find_condition(dt.start_point_, dt.image_)); 
	result.push_back(current); 
	return result; 
}

std::vector<point_info> AutomaticGenerator::getInitialPoints(Data& dt) {
	Timer search_start_timer;
	search_start_timer.Start();
	std::vector<point_info> result = findStartPoint(dt.image_);
	double endValue = search_start_timer.End(); 
	std::cout << "Time of search initial points: " << endValue << "s. " << std::endl;
	std::cout << "Initial points: " << result.size() << std::endl;
	return result; 
}

// Функция определяет наилучшую точку из sequence и добавляет её в result.
// Наилучшая точка определяется по мере Жаккара, расчитанной вдоль направления с наибольшей длиной отрезка.
// Дополнительно проверяется, что наибольшее по длине направление находится рядом со вторым по длине отрезка направлением. 
void sortSequence(const std::vector<point_info>& sequence, 
	std::vector<point_info>& result, const cv::Mat& img, double stepSize) {
	point_info best_from_sequence = point_info(cv::Point(0,0));
	double jaccard_pcur = 0;
	double best_jaccard_from_sequence = 0;
	bool isFirst = false; 
	bool anyMatch = false;
	for (const auto& pcur : sequence) {
		cv::Point c_pcur = pcur.getPoint() + pcur.getFirstDirection(); 
		std::tie(std::ignore, std::ignore, jaccard_pcur) = dda_line(c_pcur, img, pcur.getFirstDirection());
		if (jaccard_pcur >= best_jaccard_from_sequence) {
			best_from_sequence = pcur;
			best_jaccard_from_sequence = jaccard_pcur;
			isFirst = true; 
			anyMatch = true;
		}
		if (pcur.isSecondEnabled()) {
			c_pcur = pcur.getPoint() + pcur.getSecondDirection();
			std::tie(std::ignore, std::ignore, jaccard_pcur) = dda_line(c_pcur, img, pcur.getSecondDirection());
			if (jaccard_pcur >= best_jaccard_from_sequence) {
				best_from_sequence = pcur;
				best_jaccard_from_sequence = jaccard_pcur;
				isFirst = false;
				anyMatch = true;
			}
		}
	}

	if (anyMatch) {
		point_info resultPoint = point_info(best_from_sequence.getPoint());
		if (isFirst) {
			resultPoint.setMaxDirection(best_from_sequence.getFirstLength(), best_from_sequence.getFirstDirection()); 
		} else {
			resultPoint.setMaxDirection(best_from_sequence.getSecondLength(), best_from_sequence.getSecondDirection());
		}
		result.push_back(resultPoint); 
	}
	/*
	
	if (best_from_sequence.isSecondEnabled()) {
		cv::Point2f difference = best_from_sequence.getFirstDirection() - best_from_sequence.getSecondDirection(); 
		float allAbsDifference = abs(abs(difference.x) - abs(difference.y));
		if (allAbsDifference > 0.f && allAbsDifference < 2 * stepSize) {
			result.push_back(best_from_sequence);
		}
	} else {
		if (best_from_sequence.isFirstEnabled()) {
			result.push_back(best_from_sequence);
		} else {
			std::cout << "error: " << "sequence[size=" << sequence.size() << "]" << std::endl; 
		}
	}*/
}

std::vector<point_info> AutomaticGenerator::findStartPoint(cv::Mat& img)
{
	std::vector<cv::Point> contour = Data::findExternalContour(img);
	std::vector<point_info> sorted;
	LoopParameters<double> loopPars{ -1, 1, 1 };
	std::vector<point_info> result; 

	//TODO: подумать, как можно использовать нормаль для контура


	for (int contourIndex = 0; contourIndex < contour.size(); contourIndex++) {
		cv::Point2f p = contour[contourIndex]; 
		double sum = 0;
		point_info currentPoint(p);
		int maxCount = 0;
		int notNullCount = 0; 


		for (double i = loopPars.start; i <= loopPars.end; i += loopPars.step) {
			for (double j = loopPars.start; j <= loopPars.end; j += loopPars.step) {
				if (isAbsoluteSmallOf(i, 0.1f) && isAbsoluteSmallOf(j, 0.1f)) {
					continue;
				}
				cv::Point2f direction = cv::Point2f(j, i);
				cv::Point2f current = p + direction; 
				if (isPointInside(cv::Point(current), img)) {
					double length = distanceFromPointByDirection(current, direction, img);
					if (length >= 1.0) {
						currentPoint.setMaxDirection(length, resize(direction, 1.0));
						sum += length;
						notNullCount++; 
					}
				}
				maxCount++; 
			}
		}

		if (notNullCount > (maxCount / 2)) {
			continue;
		}

		if (currentPoint.isSecondEnabled() && 
			cos_about_two_vectors(currentPoint.getFirstDirection(), currentPoint.getSecondDirection()) < 0
		) {
			continue;
		}

		if (currentPoint.isFirstEnabled() && sum > 0.0f && 2 * currentPoint.getFirstLength() > sum) {
			sorted.push_back(currentPoint);
		}			
	}

	std::vector<point_info> current;

	for (const auto& p : sorted) {
		if (!(current.empty() || euclid_dst(p.getPoint(), (current.end() - 1)->getPoint()) < 20.0)) {
			sortSequence(current, result, img, loopPars.step);
			current.clear();
		}
		current.push_back(p);
	}

	if (!current.empty()) {
		sortSequence(current, result, img, loopPars.step);
	}

#ifdef DEBUG
	cv::Mat result = cv::Mat::zeros(img.size(), CV_8UC3);
	result.setTo(cv::Scalar(255, 255, 255), img != 0);
	for (auto p : corrected1) {
		cv::circle(result, p.point, 5, cv::Scalar(0, 255, 0), cv::FILLED);
	}
	myImshow(result, 0.0);
#endif // DEBUG
	return result;
}

int getRadius(std::vector<double> values, int index) {
	double v = values[index]; 
	if (v < 0.7) {
		return 1;
	}
	else if (v < 0.8) {
		return 2;
	}
	else if (v < 0.95) {
		return 3;
	}
	else return 4;
}



void drawIntermediateImages(
	std::vector<cv::Point2f> centers, 
	cv::Mat originalImg, 
	cv::Mat imgWithOnlyContour, 
	ContourIndexer* externalContourIndexer,
	std::vector<double> localJaccards
) {
	cv::Mat dividingLinesImg = cv::Mat::zeros(originalImg.size(), CV_8UC3);
	dividingLinesImg.setTo(cv::Scalar(255, 255, 255), originalImg == 0);
	
	cv::Mat centersImg = cv::Mat::zeros(originalImg.size(), CV_8UC3);
	centersImg.setTo(cv::Scalar(255, 255, 255), originalImg == 0);

	for (size_t i = 0; i < centers.size(); i++) {
		auto center = centers[i];
		auto normal = (i == 0) ? (centers[i + 1] - center) : (center - centers[i - 1]);
		auto indexesPair = getIndexesByDirection(normal, center, imgWithOnlyContour, *externalContourIndexer);

		cv::line(
			dividingLinesImg,
			externalContourIndexer->get(indexesPair.first),
			externalContourIndexer->get(indexesPair.second),
			cv::Scalar(0, 255, 0)
		);
		cv::circle(centersImg, center, getRadius(localJaccards, i), cv::Scalar(0, 255, 255), cv::FILLED);
	}

	cv::imwrite("d1.bmp", dividingLinesImg);
	cv::imwrite("d2.bmp", centersImg);
}


void CurvilinearSymmetryBase::processing(Data& dt, Curvilinear& line) {
	contour = dt.external_contour_;

	full_timer.Start();
	std::vector<point_info> start_points_ = dt.definedInitialPoint ? (UserInputGenerator().getInitialPoints(dt)) 
																   : (AutomaticGenerator().getInitialPoints(dt));
	std::vector<double> bestLocalJaccard; 

	imgWithOnlyContour = cv::Mat::zeros(dt.image_.size(), dt.image_.type());
	std::vector<std::vector<cv::Point>> contours; 
	contours.push_back(contour); 
	cv::drawContours(imgWithOnlyContour, contours, -1, 255, cv::FILLED);
	externalContourIndexer = new ContourIndexer(imgWithOnlyContour, contour); 

	for (const auto& p : start_points_) {
		Curvilinear current(double(dt.area_));

		dt.start_point_ = p.getPoint();
		current.curvilinearAxisAtoms = algorithm(dt, p.getFirstDirection());
		if (current.curvilinearAxisAtoms.size() < 2) {
			continue; 
		}
#ifdef DRAW
		cv::Mat part = cv::Mat(dt.image_.size(), CV_8UC3);
		part.setTo(cv::Scalar(255, 255, 255), dt.image_ == 0);
		for (size_t i = 0; i < indexes.size(); i++) {
			auto indexPair = indexes[i]; 
			cv::line(part, contour[indexPair.first], contour[indexPair.second], cv::Scalar(0, 255, 0));
		}
			
		for (size_t i = 1; i < current.straight_.size(); i++) {
			cv::line(part, current.straight_[i - 1], current.straight_[i], cv::Scalar(0, 255, 255));
		}
			
		cv::circle(part, current.straight_[0], 2, cv::Scalar(0, 0, 255), cv::FILLED);
		cv::imwrite("parts_m.bmp", part);

#endif // DRAW


		Timer cut_timer;

		cut_timer.Start();
		current.rectified_image_ = Cut(dt.image_, current.curvilinearAxisAtoms, contour, current.line_of_rectified_);
		current.cut_time_ = cut_timer.End(); 

		if (current.rectified_image_.size().area() < dt.area_)
			continue;
		current.calculateParameters(); 

		if (current.jaccard_index_with_straightened_shape_area_ > line.jaccard_index_with_straightened_shape_area_) {
			line = current;
			bestLocalJaccard = localJaccard; 
		}
	}
	std::cout << "Full time: " << full_timer.End() << "s." << std::endl;



	std::for_each(start_points_.begin(), start_points_.end(), [&line](point_info info) { line.start_points_.push_back(info.getPoint()); });
	if (line.curvilinearAxisAtoms.empty() || round(line.original_shape_area_ / line.straightened_shape_area_) != 1.0) {
		std::cout << "no axis" << std::endl; 
	}
	else {
		//drawIntermediateImages(line.curvilinearAxisAtoms, dt.image_, imgWithOnlyContour, externalContourIndexer, bestLocalJaccard);
		writeResult(dt, line);
		dt.saveResultImages(line);
	}
	delete externalContourIndexer;
}