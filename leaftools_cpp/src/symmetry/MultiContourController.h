#pragma once
#include "./Data.h"
#include "./Utils.h"


class RectifyContourContainer {
protected:
	std::vector<cv::Point> contour;
	std::vector<int> rectifyMask;
	std::vector<cv::Point> rectifiedContour;


public:
	RectifyContourContainer(std::vector<cv::Point> contour);
	// »ндекс повернутой точки или -1, если не удалось распр€мить.
	int rectifyPoint(cv::Point point, double angle, cv::Point oldCenter, cv::Point newCenter);
	// ¬озвращает повернутую точку по индексу, полученному из метода rectifyPoint
	cv::Point getRectified(int index) {
		return rectifiedContour[index]; 
	}

	// ƒополн€ет пробелы в rectifiedContour между точками,
	// так как при процессе распр€млени€ могут быть большие пробелы.
	// ƒополнительно обновл€ет смещение всех точек. 
	void resize(int& min_y, int& min_x);

	void draw(cv::Mat& img, cv::Scalar color);

	void applyOffset(int min_y, int min_x) {
		for (size_t i = 0; i < rectifiedContour.size(); i++) {
			rectifiedContour[i].x -= min_x;
			rectifiedContour[i].y -= min_y;
		}
	}


	// –азмер исходного контура
	int size() {
		return contour.size(); 
	}

	// ¬озвращает элемент исходного контура по номеру index
	cv::Point operator[](size_t index) {
		return contour[index]; 
	}
};


class MultiContourImageRectifier
{
private:
	std::vector<cv::Vec4i> hierarchy;
	std::vector<RectifyContourContainer*> contours;

	// —оответствие точки на изображении к контуру
	std::map<cv::Point, std::vector<RectifyContourContainer*>, PointComparator> pointToContour;
	std::vector<cv::Point> externalContour; 

	cv::Mat imgWithContours; 

	int min_x = 0;
	int max_x = 0;
	int min_y = 0;
	int max_y = 0;


	void rectifyPoint(RectifyContourContainer* contour, cv::Point point, double angle, cv::Point rotateCenter, cv::Point newCenter); 
public:
	~MultiContourImageRectifier(); 
	MultiContourImageRectifier(const cv::Mat& img);
	void rectifySector(std::pair<int, int> currentLine, std::pair<int, int> previousLine, double angle, cv::Point rotateCenter, cv::Point newCenter);
	void rectifyTerminalSector(std::pair<int, int> terminalLine, std::pair<int, int> previousLine, double angle, cv::Point rotateCenter, cv::Point newCenter);
	void rectifyOther(double angle, cv::Point rotateCenter, cv::Point newCenter);

	cv::Mat drawRectified(); 

	// ¬озвращает смещение координат на сгенерированном изображении. 
	cv::Point getImageOffset() {
		return cv::Point(min_x, min_y); 
	}
};