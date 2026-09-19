#pragma once
using namespace std;
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <filesystem>
#include "Utils.h"


class Line
{
private:
	float measure_ = 0.0f;
	cv::Point2f first_, second_;
public:
	Line() { measure_ = 0.0; };
	~Line() {};
	float getMeasure() { return measure_; };
	void setLine(cv::Point2f first, cv::Point2f second);
	void setLine(cv::Point2f first, cv::Point2f second, float getMeasure);
	cv::Point2f getFirstPoint() { return first_; };
	cv::Point2f getSecondPoint() { return second_; };

	friend std::ostream& operator<<(std::ostream& os, const Line& current);
};


class Curvilinear {
private:
	Timer fullTimer;
public:
	std::vector<cv::Point2f> curvilinearAxisAtoms;

	cv::Mat rectified_image_;

	cv::Mat straightened_;
	cv::Mat reflected_;
	cv::Mat curvilinear_axis_;
	Line line_of_rectified_;


	float intersection_area_; 
	float straightened_shape_area_, original_shape_area_; 
	float jaccard_index_with_straightened_shape_area_, jaccard_index_with_original_shape_area_;

	double cut_time_; 
	double algorithm_time_ = 0; 

	std::vector<cv::Point> start_points_; 

	friend std::ostream& operator<<(std::ostream& os, const Curvilinear& current);

	Curvilinear() {
		original_shape_area_ = 0;
		fullTimer.Start();
	}
	Curvilinear(double originalShapeArea) {
		original_shape_area_ = originalShapeArea;
		fullTimer.Start(); 
	};

	void calculateParameters();
};


class Data
{
public:
	Data();
	int readInputData(int argc, char* argv[]);
	int takeInfoFromImage();
	virtual ~Data() {};
	static std::vector<cv::Point> findExternalContour(const cv::Mat& image_);
	void saveResultImages(const Curvilinear& line);
public:
	int area_;
	cv::Mat image_;
	std::vector<cv::Point> external_contour_;
	std::string in_path_, out_path_; 

	cv::Point2f start_point_;

	bool definedInitialPoint; 
	cv::Point offset_;
	double lambda_; 

	double stepLength; 
};