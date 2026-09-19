#pragma once
#include <opencv2/opencv.hpp>

const double PI = atan(1.0) * 4; 


inline int sign(double value) {
	return value >= 0.f ? 1 : -1;
}

inline bool isInImage(int x, int y, const cv::Mat& img) {
	return (y >= 0 && y < img.rows&& x >= 0 && x < img.cols);
}

inline bool isPointInside(const cv::Point point, const cv::Mat& img) {
	return isInImage(point.x, point.y, img) && img.at<uchar>(point) != 0;
}

inline double euclid_dst(const cv::Point2f& c_mass, const cv::Point2f& p)
{
	return (sqrt(pow(c_mass.x - p.x, 2) + pow(c_mass.y - p.y, 2)));
}

inline double jaccad(double intersection, double area) {
	return intersection / (2 * area - intersection); 
}

inline cv::Point2f resize(cv::Point2f current, double length) {
	return current / (sqrt(pow(current.x, 2) + pow(current.y, 2)) / length);
}
inline float cos_about_two_vectors(const cv::Point2f& first, const cv::Point2f& second) {
	return (first.x * second.x + first.y * second.y) / (sqrt(pow(first.x, 2) + pow(first.y, 2)) *
		sqrt(pow(second.x, 2) + pow(second.y, 2)));
}


inline bool isAbsoluteSmallOf(double value, double eps) {
	return abs(value) < abs(eps);
}

inline double calculateDistanceFromPointToLine(cv::Point2f point, cv::Point2f l1, cv::Point2f l2) {
	cv::Point2f difference = l2 - l1;
	return (difference.y * point.x - difference.x * point.y + l2.x * l1.y - l2.y * l1.x) /
		sqrt(difference.y * difference.y + difference.x * difference.x);
}

inline bool isInInterval(int value, int min, int max) {
	return std::min(min, max) <= value && value <= std::max(min, max);
}

inline bool isInInterval(double value, double limit1 , double limit2) {
	return std::min(limit1, limit2) <= value && value <= std::max(limit1, limit2);
}

inline double length(cv::Point2f direction) {
	return euclid_dst(cv::Point2f(0, 0), direction);
}


class Timer
{
	std::chrono::steady_clock::time_point begin;
public:
	Timer() { };
	void Start() {
		begin = std::chrono::steady_clock::now();
	};
	double End() {
		auto end = std::chrono::steady_clock::now();
		return std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() * 1e-6;
	}
};

bool isTwoLinesIntersection(cv::Point2f l1, cv::Point2f l2, cv::Point2f l3, cv::Point2f l4);


//TODO: переделать данный метод: Алгоритм должен работать с вещественными числами.
// Метод возвращает список направлений из center, формирующих окружность.
// Основа DDA-алгоритма.
std::vector<cv::Point2f> getCircleDirections(double radius);

std::vector<cv::Point2f> getCorrectCircleDirections(
	double radius,
	double alpha,
	cv::Point2f defaultDirection
);

cv::Point2f ddaBase(cv::Point2f center, const cv::Mat& img, cv::Point2f direction);

inline double distanceFromPointByDirection(cv::Point2f point, cv::Point2f direction, const cv::Mat& img) {
	return euclid_dst(point, ddaBase(point, img, direction));
}

std::tuple<cv::Point2f, cv::Point2f, float> dda_line(cv::Point2f center, const cv::Mat& img, cv::Point2f direction);


struct PointComparator {
	bool operator()(const cv::Point& a, const cv::Point& b) const {
		return (a.x < b.x) || (a.x == b.x && a.y < b.y);
	}
};


class ContourIndexer {
private:
	cv::Mat imgWithContour;
	std::map<cv::Point, size_t, PointComparator> mapPointToIdx;
	std::vector<cv::Point> contour; 

public:
	ContourIndexer(const cv::Mat& img, const std::vector<cv::Point>& contour) : contour(contour) {
		imgWithContour = cv::Mat::zeros(img.size(), CV_8UC1);
		cv::Point p;
		for (size_t idx = 0; idx < contour.size(); idx++) {
			p = contour[idx];
			imgWithContour.at<uchar>(p) = 255;
			mapPointToIdx.insert({ p, idx });
		}
	}

	size_t get(const cv::Point& point) const {
		return isInImage(point.x, point.y, imgWithContour) && imgWithContour.at<uchar>(point) ? mapPointToIdx.at(point) : -1;
	}

	cv::Point get(int index) {
		return contour[index]; 
	}

	int findContourPoint(int x, int y) const {
		cv::Point p;
		size_t result = -1;
		for (int i = -1; i <= 1; i++) {
			for (int j = -1; j <= 1; j++) {
				p.x = x + i;
				p.y = y + j;

				result = get(p);
				if (result != -1) {
					return result;
				}
			}
		}
		return -1;
	}
};

// Метод пускает лучи перпендикулярно направлению direction из center и возвращает индексы отрезков по данным лучам. 
std::pair<int, int> getIndexesByDirection(
	cv::Point2f direction,
	cv::Point2f center,
	const cv::Mat& filledExternalContourImg,
	const ContourIndexer& indexer
);

inline bool isTwoLinesIntersection(
	std::pair<cv::Point2f, cv::Point2f> first,
	std::pair<cv::Point2f, cv::Point2f> second
) {
	return isTwoLinesIntersection(first.first, first.second, second.first, second.second);
}


inline bool isAnyIntersection(std::vector<std::pair<cv::Point2f, cv::Point2f>> lines,
	std::pair<cv::Point2f, cv::Point2f> currentLine
) {
	for (const auto& line : lines) {
		if (isTwoLinesIntersection(line, currentLine)) {
			return true;
		}
	}
	return false;
}