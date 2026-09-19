#include "Data.h"
#include "Utils.h"

std::vector<cv::Point> Data::findExternalContour(const cv::Mat& image_)
{
	std::vector<cv::Vec4i> hierarchy;
	std::vector<std::vector<cv::Point> > AllContours;
	cv::findContours(image_, AllContours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point(0, 0));
	if (AllContours.size() == 0)
		return std::vector<cv::Point>();
	std::vector<cv::Point> result = *std::max_element(AllContours.begin(), AllContours.end(), []
	(const std::vector<cv::Point>& first, const std::vector<cv::Point>& second) { return first.size() < second.size(); });
	return result;
}

Data::Data()
{
	area_ = 0;
	start_point_.x = start_point_.y = -1; 
}

int Data::readInputData(int argc, char* argv[])
{
	stepLength = 0;
	definedInitialPoint = false; 
	if (argc < 4) {
		return -1;
	}

	{
		std::string str(argv[1]);
		str = str.substr(str.length() - 3);
		if (str != "bmp")
		{
			std::cout << "Bad file extension. Need a BMP format" << std::endl;
			return -3;
		}
	}
	in_path_ = argv[1];
	//Check image
	{
		image_ = cv::imread(in_path_, -1);
		if ((image_.data == NULL))
		{
			std::cout << "Non corrent file name" << std::endl;
			return -21;
		}
		if ((image_.channels() != 1))
		{
			std::cout << "Not binary img" << std::endl;
			return -22;
		}
		cv::bitwise_not(image_, image_);
		area_ = cv::countNonZero(image_);
		if (area_ == image_.cols * image_.rows || area_ == 0)
		{
			std::cout << "Empty img" << std::endl;
			return -23;
		}
	}
	out_path_ = "";
	int current = 2;

	bool l_flag = false, x_flag = false, y_flag = false; 

	for (int i = current; i < argc; i += 2) {
		if (strcmp(argv[i], "-l") == 0) {
			lambda_ = std::stod(argv[i + 1]);
			if (lambda_ < 0 || lambda_ > 1.0) {
				std::cout << "Bad agrument of lambda" << std::endl;
				return -5;
			}
			l_flag = true; 
			continue;
		}
		if (strcmp(argv[i], "-x") == 0) {
			start_point_.x = std::stoi(argv[i + 1]);
			if (start_point_.x < 0 || start_point_.x >= image_.cols )
			{
				std::cout << "Bad X value." << std::endl;
				return -8;
			}
			x_flag = true; 
			continue;
		}
		if (strcmp(argv[i], "-y") == 0) {
			start_point_.y = std::stoi(argv[i + 1]);
			if (start_point_.y < 0 || start_point_.y >= image_.rows)
			{
				std::cout << "Bad Y value." << std::endl;
				return -8;
			}
			y_flag = true; 
			continue;
		}
		if (strcmp(argv[i], "-d") == 0) {
			if (!std::filesystem::is_directory(argv[i + 1])) {
				std::cout << "Bad result directory value." << std::endl;
				return -8;
			}
			else {
				out_path_ = argv[i + 1];
			}
			continue;
		}

		if (strcmp(argv[i], "-h") == 0) {
			stepLength = std::stod(argv[i + 1]);
			continue; 
		}
	}

	if (!l_flag) {
		std::cout << "Lambda is not defined." << std::endl;
		return -3;
	}

	if (x_flag && y_flag && image_.at<uchar>(start_point_) == 0)
	{
		std::cout << "Bad start point." << std::endl;
		return -3;
	}

	if (x_flag && y_flag)
		definedInitialPoint = true;

	return 0;
}

int Data::takeInfoFromImage()
{
	offset_.x = offset_.y = 0;
	//delete empty rows and empty cols
	{
		cv::Rect rect = cv::boundingRect(image_);
		offset_.x = rect.x;
		offset_.y = rect.y;
		image_ = image_(cv::Range(rect.y, rect.y + rect.height), cv::Range(rect.x, rect.x + rect.width));
	}
	start_point_.x -= offset_.x;
	start_point_.y -= offset_.y; 
	external_contour_ = findExternalContour(image_);
	if (external_contour_.size() == 0)
		return -1;
	image_.setTo(1, (image_ != 0));
	return 0;
}



std::ostream& operator<<(std::ostream& os, const Curvilinear& current)
{
	os << "Straightening time: " << current.cut_time_ << "s." << std::endl;
	std::cout << "Algorithm Simple Time: " << current.algorithm_time_ << "s." << std::endl;
	os << "J(A)=" << current.jaccard_index_with_straightened_shape_area_ << std::endl;
	os << "J(A')=" << current.jaccard_index_with_original_shape_area_ << std::endl;
	os << "A (rectified) = " << current.straightened_shape_area_ << std::endl;
	os << "A' (original) = " << current.original_shape_area_ << std::endl;
	os << "Number of segments is " << current.curvilinearAxisAtoms.size() - 1;
	return os;
}


void Data::saveResultImages(const Curvilinear& line)
{
	if (line.curvilinearAxisAtoms.empty()) {
		return; 
	}
	// Если путь не задан (режим модуля, не CLI) — не пишем файлы.
	if (in_path_.empty()) {
		return;
	}
	size_t pos = 0;
	pos = in_path_.rfind("\\") + 1;
	if (pos > in_path_.length())
		pos = 0;
	std::string fNameOutBMP = out_path_ + in_path_.substr(pos);
	pos = fNameOutBMP.rfind(".");
	std::string buffer = fNameOutBMP.substr(0, pos);
	std::string end = fNameOutBMP.substr(pos);

	std::string straightened_file_name = buffer + "_straightened" + end;
	std::string reflected_file_name = buffer + "_reflected" + end;
	std::string axis_file_name = buffer + "_curvilinear_axis" + end;

	std::cout << straightened_file_name << std::endl;
	std::cout << reflected_file_name << std::endl;
	std::cout << axis_file_name << std::endl;
	cv::imwrite(straightened_file_name, line.straightened_); 
	cv::imwrite(reflected_file_name, line.reflected_); 
	cv::imwrite(axis_file_name, line.curvilinear_axis_);
}

void Line::setLine(cv::Point2f first, cv::Point2f second)
{
	cv::Point2f zero(0, 0);
	if (first.y > second.y || first.x > second.x && first.y == second.y)
	{
		std::swap(first, second);
	}
	measure_ = 0.0;
	first_ = first;
	second_ = second;
}


void Line::setLine(cv::Point2f first, cv::Point2f second, float getMeasure)
{
	setLine(first, second);
	measure_ = getMeasure;
}


std::ostream& operator<<(std::ostream& os, const Line& current)
{
	os << current.first_.x << "," << current.first_.y << ","
	   << current.second_.x << "," << current.second_.y << ","
	   << current.measure_;
	return os;
}

void pruning(cv::Mat& src, cv::Mat& dst, cv::Point oldP, cv::Point newP)
{
	int w = std::min(newP.x, oldP.x);
	int h = std::min(newP.y, oldP.y);
	int h1 = std::min(src.rows - oldP.y, dst.rows - newP.y);
	int w1 = std::min(src.cols - oldP.x, dst.cols - newP.x);
	src = src(cv::Range(oldP.y - h, oldP.y + h1), cv::Range(oldP.x - w, oldP.x + w1));
	dst = dst(cv::Range(newP.y - h, newP.y + h1), cv::Range(newP.x - w, newP.x + w1));
}

double calcIntersection(cv::Mat src, Line& axis)
{
	cv::Point2f first = axis.getFirstPoint();
	cv::Point2f second = axis.getSecondPoint();
	if (first.y > second.y || first.x > second.x && first.y == second.y)
	{
		std::swap(first, second);
	}
	cv::Mat src_copy(src.size(), src.type());
	src.copyTo(src_copy);
	cv::Mat dst = cv::Mat::zeros(src.size(), src.type());
	if (abs(first.x - second.x) == abs(first.y - second.y))
	{
		// for 45 degrees 
		if (first.y < second.y && first.x > second.x)
		{
			std::swap(first, second);
		}

		if (first.y > second.y && first.x < second.x)
		{
			cv::flip(src_copy, src_copy, 0);

			first.y = src_copy.rows - first.y - 1;
			second.y = src_copy.rows - second.y - 1;
		}

		cv::Point2f new_first = cv::Point2f(first.y, first.x);
		cv::Point2f new_second = cv::Point2f(second.y, second.x);
		cv::transpose(src_copy, dst);

		pruning(src_copy, dst, cv::Point((int)first.x, (int)first.y), cv::Point((int)new_first.x, (int)new_first.y));
	}
	else
	{
		if (first.x == second.x)
		{
			cv::flip(src_copy, dst, 1);
			// for vertical line

			float newX = src_copy.cols - first.x - 1;
			if (newX - first.x >= 0)
			{
				dst = dst(cv::Range(0, dst.rows), cv::Range((int)(newX - first.x), dst.cols));
				src_copy = src_copy(cv::Range(0, src_copy.rows), cv::Range(0, src_copy.cols - (int)(newX - first.x)));
			}
			else
			{
				dst = dst(cv::Range(0, dst.rows), cv::Range(0, dst.cols + (int)(newX - first.x)));
				src_copy = src_copy(cv::Range(0, src_copy.rows), cv::Range((int)(first.x - newX), src_copy.cols));
			}
		}
		else
		{
			if (first.y == second.y)
			{
				cv::flip(src_copy, dst, 0);
				// for horizontal line
				float newY = src_copy.rows - first.y - 1;
				if (newY - first.y >= 0)
				{
					dst = dst(cv::Range((int)(newY - first.y), dst.rows), cv::Range(0, dst.cols));
					src_copy = src_copy(cv::Range(0, src_copy.rows - (int)(newY - first.y)), cv::Range(0, src_copy.cols));
				}
				else
				{
					dst = dst(cv::Range(0, dst.rows + (int)(newY - first.y)), cv::Range(0, dst.cols));
					src_copy = src_copy(cv::Range(int(first.y - newY), src_copy.rows), cv::Range(0, src_copy.cols));
				}
			}
			else
			{
				double k = double(second.y - first.y) / double(second.x - first.x);
				double b = first.y - k * first.x;
				double axis_cos = sqrt(1.0 / (1.0 + pow(k, 2)));
				double axis_sin = k * axis_cos;
				{
					// for other lines
					cv::Mat matrix = cv::Mat::zeros(2, 3, CV_64FC1);
					matrix.at<double>(0, 0) = axis_cos * axis_cos - axis_sin * axis_sin;
					matrix.at<double>(0, 1) = 2 * axis_cos * axis_sin;
					matrix.at<double>(0, 2) = -2 * b * axis_cos * axis_sin;
					matrix.at<double>(1, 0) = 2 * axis_cos * axis_sin;
					matrix.at<double>(1, 1) = -axis_cos * axis_cos + axis_sin * axis_sin;
					matrix.at<double>(1, 2) = 2 * b * axis_cos * axis_cos;
					cv::warpAffine(src, dst, matrix, src.size(), cv::INTER_LINEAR);
				}
			}
		}
	}
	cv::Mat intersection;
	cv::bitwise_and(src_copy, dst, intersection);
	return cv::countNonZero(intersection);
}


void Curvilinear::calculateParameters() {
	intersection_area_ = calcIntersection(rectified_image_, line_of_rectified_);
	straightened_shape_area_ = cv::countNonZero(rectified_image_);
	jaccard_index_with_straightened_shape_area_ = jaccad(intersection_area_, straightened_shape_area_);
	jaccard_index_with_original_shape_area_ = jaccad(intersection_area_, original_shape_area_);
	algorithm_time_ = fullTimer.End(); 
}