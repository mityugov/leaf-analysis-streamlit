#pragma once
#include "./Utils.h"
#include <numeric>


struct Atom {
	cv::Point2f point; 
	std::pair<cv::Point2f, cv::Point2f> contourPoints;
	std::pair<int, int> contourIndexes; 
	double localJaccard;
	cv::Point2f directionToPoint; 
};




class IterationProcessor {
	std::vector<Atom> resultAtoms; 
	std::vector<Atom> best;
	double bestEstimation = 0.0; 

public:
	cv::Mat img;
	cv::Mat filledExternalContourImage;
	ContourIndexer* indexer;
	std::vector<std::pair<cv::Point2f, cv::Point2f>> points;

	Atom startAtom;
	Atom endAtom; 

	IterationProcessor() = delete;

	IterationProcessor(
		cv::Mat& img,
		cv::Mat& filledExternalContourImage,
		ContourIndexer* indexer,
		std::vector<std::pair<cv::Point2f, cv::Point2f>> points
	) {
		this->img = img; 
		this->filledExternalContourImage = filledExternalContourImage; 
		this->indexer = indexer; 
		this->points = points; 
	}

	void prepare(
		std::pair<int, int> endLine,
		std::pair<int, int> startLine,
		cv::Point2f start,
		cv::Point2f end,
		cv::Point2f startDirection
	) {
		best.clear(); 
		resultAtoms.clear(); 
		bestEstimation = 0.0; 
		endAtom = {}; 
		endAtom.point = end; 
		endAtom.contourIndexes = endLine; 
		endAtom.contourPoints = { indexer->get(endLine.first), indexer->get(endLine.second) };
 

		startAtom = {}; 
		startAtom.point = start; 
		startAtom.localJaccard = 0.0; 
		startAtom.directionToPoint = startDirection; 
		startAtom.contourIndexes = startLine; 
		startAtom.contourPoints = { indexer->get(startLine.first), indexer->get(startLine.second) };
	}

	cv::Point2f getPreviousStepDirection() {
		return resultAtoms.empty() ? startAtom.directionToPoint :  (resultAtoms.end() - 1)->directionToPoint;
	}

	inline bool areBorderedLinesIntersectedWith(std::pair<int, int> current) {
		std::pair<cv::Point2f, cv::Point2f> currentLine;
		currentLine.first = indexer->get(current.first);
		currentLine.second = indexer->get(current.second);

		if (!points.empty() && isAnyIntersection(points, currentLine)) {
			return true; 
		}

		for (const Atom& atom : resultAtoms) {
			if (isTwoLinesIntersection(atom.contourPoints, currentLine)) {
				return true;
			}
		}

		return isTwoLinesIntersection(startAtom.contourPoints, currentLine) || isTwoLinesIntersection(endAtom.contourPoints , currentLine);
	}

	void addIteration(Atom& atom) {
		atom.contourPoints = { indexer->get(atom.contourIndexes.first), indexer->get(atom.contourIndexes.second) }; 
		resultAtoms.push_back(atom); 
	}

	void removeIteration() {
		resultAtoms.pop_back();
	}

	std::vector<Atom> getResult() {
		return best;
	}

	void saveBest() {
		double estimate = 0; 
		std::for_each(resultAtoms.begin(), resultAtoms.end(), [&estimate](Atom& atom) {estimate += atom.localJaccard; });
		estimate /= resultAtoms.size(); 
		if (estimate > bestEstimation) {
			best = resultAtoms;
			bestEstimation = estimate;
		}
	}

	std::pair<int, int> getIndexesFor(cv::Point2f newCenter, cv::Point2f direction) {
		return getIndexesByDirection(direction, newCenter, filledExternalContourImage, *indexer); 
	}
};



class Smoother
{
private:
	IterationProcessor* processor; 
public: 
	Smoother(
		cv::Mat img,
		cv::Mat filledExternalContourImage,
		ContourIndexer* indexer,
		std::vector<std::pair<cv::Point2f, cv::Point2f>> points
	) {
		processor = new IterationProcessor(
			img,
			filledExternalContourImage,
			indexer,
			points
		);
	}

	~Smoother() {
		delete processor; 
	}


	// ¬озвращает список дополнительных точек между start и end.
	std::vector<Atom> smooth(double length, std::pair<int, int> endLine,
		std::pair<int, int> startLine,
		cv::Point2f start,
		cv::Point2f end,
		cv::Point2f startDirection);
};

