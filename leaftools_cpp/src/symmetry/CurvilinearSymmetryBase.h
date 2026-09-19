#pragma once
#include "./Data.h"
#include "Utils.h"
#include "./MultiContourController.h"

template<typename T>
struct LoopParameters {
	T start; 
	T end; 
	T step;
};

class point_info {
public:
	point_info(cv::Point2f point) {
		this->point = point; 
		this->firstMaxEnabled = false; 
		this->secondMaxEnabled = false; 
	}

	cv::Point2f getPoint() const {
		return point; 
	}

	void setMaxDirection(double length, cv::Point2f direction) {
		std::pair<double, cv::Point2f> newDirection = { length, direction };
		if (!firstMaxEnabled) {
			firstMax = newDirection;
			firstMaxEnabled = true; 
			return;
		} 

		if (firstMax.first < newDirection.first) {
			std::swap(firstMax, newDirection); 
		}

		if (!secondMaxEnabled) {
			secondMax = newDirection; 
			secondMaxEnabled = true;
			return; 
		}

		if (secondMax.first < newDirection.first) {
			secondMax = newDirection; 
		}
	}

	bool isFirstEnabled() const {
		return firstMaxEnabled; 
	}

	bool isSecondEnabled() const {
		return secondMaxEnabled; 
	}

	double getFirstLength() const {
		return firstMax.first; 
	}

	double getSecondLength() const {
		return secondMax.first; 
	}

	cv::Point2f getFirstDirection() const {
		return firstMax.second; 
	}

	cv::Point2f getSecondDirection() const {
		return secondMax.second; 
	}

private:
	cv::Point2f point;
	std::pair<double, cv::Point2f> firstMax, secondMax;
	bool firstMaxEnabled, secondMaxEnabled;
};



class InitialPointsGenerator {
public:
	InitialPointsGenerator() {};
	virtual std::vector<point_info> getInitialPoints(Data& dt) = 0;
	virtual ~InitialPointsGenerator() {};
};

class UserInputGenerator : public InitialPointsGenerator {
public:
	UserInputGenerator() {};
	~UserInputGenerator() {};
	std::vector<point_info> getInitialPoints(Data& dt);
};

class AutomaticGenerator : public InitialPointsGenerator {
public:
	AutomaticGenerator() {};
	~AutomaticGenerator() {};
	std::vector<point_info> getInitialPoints(Data& dt);
private:
	std::vector<point_info> findStartPoint(cv::Mat& img);
};

class CurvilinearSymmetryBase
{
public:
	CurvilinearSymmetryBase() {}; 
	virtual ~CurvilinearSymmetryBase() {};
	void processing(Data& dt, Curvilinear& line);
protected:
	virtual std::vector<cv::Point2f> algorithm(Data& dt, cv::Point2f start_condition) = 0;
	void writeResult(Data& dt, Curvilinear& line);
	cv::Point2f find_condition(cv::Point2f center, cv::Mat& img);


	std::vector<point_info> find_start_point(cv::Mat& img);
	std::vector<cv::Point> contour;
	Timer full_timer;
	cv::Mat Cut(const cv::Mat& img, 
		const std::vector<cv::Point2f>& centers,
		const std::vector<cv::Point>& contour,
		Line& line_of_rectified_);
	void Draw(Data& dt, Curvilinear& line);

	cv::Mat imgWithOnlyContour; 
	ContourIndexer* externalContourIndexer; 
};


class GreedySearch : public CurvilinearSymmetryBase
{
public:
	GreedySearch() {};
	~GreedySearch() {};
private:
	// Возвращает направление для смещения и индексы внешнего контура для разреза фигуры. 
	std::tuple<cv::Point2f, std::pair<int, int>>  findNextDirection(
		cv::Point2f center,
		cv::Point2f last_condition,
		cv::Mat img,
		double h,
		float alpha_degrees,
		double lambda,
		bool isFirstCenter,
		std::pair<int, int> prevStepLine,
		const std::vector<std::pair<cv::Point2f, cv::Point2f>> previousLines);
	std::vector<cv::Point2f> algorithm(Data& dt, cv::Point2f start_condition);
};





