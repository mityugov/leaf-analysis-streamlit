#include "rotation_profile.h"

#include <algorithm>
#include <cmath>

namespace {

inline double JACCARD(double intersection, double area)
{
    return intersection / (2 * area - intersection);
}

} // namespace

std::map<double, double> generateDescriptorSequence(cv::Mat src, cv::Point center, int area)
{
	cv::Mat img90, img180, img270;// , original_;
	cv::Mat img45, img135, img225, img315;
	std::map<double, double> result;
	double max_distance = cv::norm(cv::Point(0, 0) - center);
	max_distance = std::max(max_distance, cv::norm(cv::Point(src.cols - 1, 0) - center));
	max_distance = std::max(max_distance, cv::norm(cv::Point(0, src.rows - 1) - center));
	max_distance = std::max(max_distance, cv::norm(cv::Point(src.cols - 1, src.rows - 1) - center));

	//findMaxDistance(contour, center, max_distance);
	max_distance = round(max_distance);

	cv::Mat updated_src;
	cv::Point updated_center;

	{
		int up = center.y - max_distance;
		if (up > 0)
			up = 0;
		int down = (src.rows - center.y) - max_distance;
		if (down > 0)
			down = 0;
		int left = center.x - max_distance;
		if (left > 0)
			left = 0;
		int right = (src.cols - center.x) - max_distance;
		if (right > 0)
			right = 0;
		up = abs(up);
		down = abs(down);
		left = abs(left);
		right = abs(right);
		updated_src = cv::Mat::zeros(up + src.rows + down, left + src.cols + right, src.type());
		src.copyTo(updated_src(cv::Range(up, up + src.rows), cv::Range(left, left + src.cols)));
		updated_center = cv::Point(center.x + left, center.y + up);
	}


	{
		cv::warpAffine(updated_src, img45, cv::getRotationMatrix2D(updated_center, 45, 1.0), updated_src.size(), cv::INTER_LINEAR);
		cv::rotate(img45, img135, cv::ROTATE_90_COUNTERCLOCKWISE);
		cv::rotate(img45, img225, cv::ROTATE_180);
		cv::rotate(img45, img315, cv::ROTATE_90_CLOCKWISE);
	}
	cv::rotate(updated_src, img90, cv::ROTATE_90_COUNTERCLOCKWISE);
	cv::rotate(updated_src, img180, cv::ROTATE_180);
	cv::rotate(updated_src, img270, cv::ROTATE_90_CLOCKWISE);
	cv::Point center90 = cv::Point(updated_center.y, updated_src.cols - updated_center.x - 1);
	cv::Point center180 = cv::Point(updated_src.cols - updated_center.x - 1, updated_src.rows - updated_center.y - 1);
	cv::Point center270 = cv::Point(updated_src.rows - updated_center.y - 1, updated_center.x);




	cv::Mat iteration_dst;
	//cv::Mat iteration_mem;
	//cv::Mat src_mem;
	//updated_src.copyTo(src_mem);

	cv::Mat buf_inter;
	int count_black_inter;

	int h, w, h1, w1;
	cv::Mat src_prunned, dst_prunned;


	for (int i = 0; i <= 22; i++)
	{
		// 0-22
		if (i == 0)
		{
			result.insert({ 0, 1.0 });
			updated_src.copyTo(iteration_dst);
		}
		else
		{
			cv::warpAffine(updated_src, iteration_dst,
				cv::getRotationMatrix2D(updated_center, i, 1.0),
				updated_src.size(), cv::INTER_LINEAR);
			cv::bitwise_and(updated_src, iteration_dst, buf_inter);
			count_black_inter = cv::countNonZero(buf_inter);
			result.insert({ i, JACCARD(count_black_inter, area) });
		}
		//iteration_dst.copyTo(iteration_mem);

		//23-45
		cv::bitwise_and(iteration_dst, img45, buf_inter);
		count_black_inter = cv::countNonZero(buf_inter);
		//std::cout << JACCARD(count_black_inter, area) << std::endl;
		result.insert({ 45 - i, JACCARD(count_black_inter, area) });


		if (i != 0)
		{
			//46-68
			//img315.copyTo(src_mem);
			//pruning(iteration_dst, src_mem, updated_center, center270);
			w = std::min(updated_center.x, center270.x);
			h = std::min(updated_center.y, center270.y);
			h1 = std::min(iteration_dst.rows - updated_center.y, img315.rows - center270.y);
			w1 = std::min(iteration_dst.cols - updated_center.x, img315.cols - center270.x);
			src_prunned = iteration_dst(cv::Range(updated_center.y - h, updated_center.y + h1),
				              cv::Range(updated_center.x - w, updated_center.x + w1));
			dst_prunned = img315(cv::Range(center270.y - h, center270.y + h1), cv::Range(center270.x - w, center270.x + w1));
			cv::bitwise_and(src_prunned, dst_prunned, buf_inter);
			count_black_inter = cv::countNonZero(buf_inter);
			result.insert({ 45 + i, JACCARD(count_black_inter, area) });
			//iteration_mem.copyTo(iteration_dst);
			//result.insert({ 45 + i, jaccard_calculate(iteration_dst, img315, area) });

			//91-113
			//img270.copyTo(src_mem);
			//pruning(iteration_dst, src_mem, updated_center, center270);
			src_prunned = iteration_dst(cv::Range(updated_center.y - h, updated_center.y + h1),
				cv::Range(updated_center.x - w, updated_center.x + w1));
			dst_prunned = img270(cv::Range(center270.y - h, center270.y + h1), cv::Range(center270.x - w, center270.x + w1));
			cv::bitwise_and(src_prunned, dst_prunned, buf_inter);
			count_black_inter = cv::countNonZero(buf_inter);
			result.insert({ 90 + i, JACCARD(count_black_inter, area) });
			//iteration_mem.copyTo(iteration_dst);

			//114-135
			//img135.copyTo(src_mem);
			//pruning(iteration_dst, src_mem, updated_center, center90);
			w = std::min(updated_center.x, center90.x);
			h = std::min(updated_center.y, center90.y);
			h1 = std::min(iteration_dst.rows - updated_center.y, img135.rows - center90.y);
			w1 = std::min(iteration_dst.cols - updated_center.x, img135.cols - center90.x);
			src_prunned = iteration_dst(cv::Range(updated_center.y - h, updated_center.y + h1),
				cv::Range(updated_center.x - w, updated_center.x + w1));
			dst_prunned = img135(cv::Range(center90.y - h, center90.y + h1), cv::Range(center90.x - w, center90.x + w1));
			cv::bitwise_and(src_prunned, dst_prunned, buf_inter);
			count_black_inter = cv::countNonZero(buf_inter);
			result.insert({ 135 - i, JACCARD(count_black_inter, area) });
			//iteration_mem.copyTo(iteration_dst);
			//result.insert({ 135 - i, jaccard_calculate(iteration_dst, img135, area) });
		}

		//69-90
		//img90.copyTo(src_mem);
		//pruning(iteration_dst, src_mem, updated_center, center90);
		w = std::min(updated_center.x, center90.x);
		h = std::min(updated_center.y, center90.y);
		h1 = std::min(iteration_dst.rows - updated_center.y, img90.rows - center90.y);
		w1 = std::min(iteration_dst.cols - updated_center.x, img90.cols - center90.x);
		src_prunned = iteration_dst(cv::Range(updated_center.y - h, updated_center.y + h1),
			cv::Range(updated_center.x - w, updated_center.x + w1));
		dst_prunned = img90(cv::Range(center90.y - h, center90.y + h1), cv::Range(center90.x - w, center90.x + w1));
		cv::bitwise_and(src_prunned, dst_prunned, buf_inter);
		//cv::bitwise_and(iteration_dst, src_mem, buf_inter);
		count_black_inter = cv::countNonZero(buf_inter);
		result.insert({ 90 - i, JACCARD(count_black_inter, area) });
		//iteration_mem.copyTo(iteration_dst);

		//136-158
		//img225.copyTo(src_mem);
		//pruning(iteration_dst, src_mem, updated_center, center180);
		w = std::min(updated_center.x, center180.x);
		h = std::min(updated_center.y, center180.y);
		h1 = std::min(iteration_dst.rows - updated_center.y, img225.rows - center180.y);
		w1 = std::min(iteration_dst.cols - updated_center.x, img225.cols - center180.x);
		src_prunned = iteration_dst(cv::Range(updated_center.y - h, updated_center.y + h1),
			cv::Range(updated_center.x - w, updated_center.x + w1));
		dst_prunned = img225(cv::Range(center180.y - h, center180.y + h1), cv::Range(center180.x - w, center180.x + w1));
		cv::bitwise_and(src_prunned, dst_prunned, buf_inter);
		//cv::bitwise_and(iteration_dst, src_mem, buf_inter);
		count_black_inter = cv::countNonZero(buf_inter);
		result.insert({ 135 + i, JACCARD(count_black_inter, area) });
		//iteration_mem.copyTo(iteration_dst);
		//result.insert({ 135 + i, jaccard_calculate(iteration_dst, img225, area) });

		//159-180
		//img180.copyTo(src_mem);
		//pruning(iteration_dst, src_mem, updated_center, center180);
		src_prunned = iteration_dst(cv::Range(updated_center.y - h, updated_center.y + h1),
			cv::Range(updated_center.x - w, updated_center.x + w1));
		dst_prunned = img180(cv::Range(center180.y - h, center180.y + h1), cv::Range(center180.x - w, center180.x + w1));
		cv::bitwise_and(src_prunned, dst_prunned, buf_inter);
		//cv::bitwise_and(iteration_dst, src_mem, buf_inter);
		count_black_inter = cv::countNonZero(buf_inter);
		result.insert({ 180 - i, JACCARD(count_black_inter, area) });
	}
	return result;
}
