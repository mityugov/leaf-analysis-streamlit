#include "./Smoother.h"

const double limitCos = 0.7; 

bool iterationProcess(
	cv::Point2f start,
	cv::Point2f end,
	double length,
	IterationProcessor& processor
) {
	double maxDistance = euclid_dst(start, end);
	if (maxDistance <= length) {
		processor.saveBest(); 
		return true;
	}

	double maxAngle = round(90.0 / std::max(log(maxDistance), 2.0));
	cv::Point2f directionToEnd = resize(end - start, length);
	double lowJaccard = 0;
	auto prevDirection = processor.getPreviousStepDirection();
	{
		cv::Point2f newCenter = start + prevDirection; ;
		if (isPointInside(newCenter, processor.img)) {
			std::tie(std::ignore, std::ignore, lowJaccard) = dda_line(newCenter, processor.img, prevDirection);
		}
		
		newCenter = start + directionToEnd; 
		if (isPointInside(newCenter, processor.img)) {
			double prevDirectionJaccard;
			std::tie(std::ignore, std::ignore, prevDirectionJaccard) = dda_line(newCenter, processor.img, directionToEnd);
			lowJaccard = std::max(prevDirectionJaccard, lowJaccard); 
		}
	}
	
	cv::Point2f newPoint;
	
	double localJaccard;
	//TODO: нужно сделать взвешенный localJaccard на cos (между либо предыдущей и следующей	точками, либо текущим направлением и следующим направлением)

	std::map<double, Atom, std::greater<double>> directionAtoms;
	for (auto direction : getCorrectCircleDirections(length, maxAngle, prevDirection)) {
		newPoint = direction + start; 
		if (!isPointInside(newPoint, processor.img)) {
			continue; 
		}

		// ƒанным ограничением не даем делать резкие заломы оси криволинейной симметрии.
		double cos = cos_about_two_vectors(direction, directionToEnd);
		if (abs(cos) < limitCos) {
			continue;
		}

		// ќграничение, чтобы не делать перескоки через фон. 
		if (distanceFromPointByDirection(start, direction, processor.img) < length) {
			continue; 
		}

		// "зрительный" контакт с конечной точкой должен быть всегда. ≈сли его нет, значит, ушли куда-то не туда.
		if (distanceFromPointByDirection(newPoint, (end - newPoint), processor.img) < euclid_dst(newPoint, end)) {
			continue;
		}

		// Ќельз€, чтобы полученный отрезок разделени€ пересекалс€ с другой точкой.
		std::pair<int, int> currentLineIndexes = processor.getIndexesFor(newPoint, direction);
		if (processor.areBorderedLinesIntersectedWith(currentLineIndexes)) {
			continue;
		}
		
		std::tie(std::ignore, std::ignore, localJaccard) = dda_line(newPoint, processor.img, direction);
		if (localJaccard < lowJaccard) {
			continue; 
		}

		Atom atom;
		atom.localJaccard = localJaccard; 
		atom.directionToPoint = direction; 
		atom.point = newPoint;
		atom.contourIndexes = currentLineIndexes; 
		directionAtoms.insert({ localJaccard, atom }); 
	}

	for (auto& kv : directionAtoms) {
		Atom& atom = kv.second; 
		processor.addIteration(atom); 
		if (iterationProcess(atom.point, end, length, processor)) {
			return true;
		}
		processor.removeIteration();
	}
	return false;
}


// ≈сть базова€ процедура обхода, котора€ работает очень медленно. 
// ќна перебирает все возможные точки с целью поиска наилучшего фрагмента криволинейной оси из точки ј в B.
// јлгоритм - brutforce, то есть, работает по принципу полного перебора. 
// Ќужно добавить ограничени€ по выбору точек дл€ движени€. 
// 
// 1.  оличество углов поворота зависит от максимального рассто€ни€ от start по направлению движени€. 
//    Ёто позвол€ет уменьшить на каждой итерации количество углов дл€ просмотра. 
// 
// Ќужно придумать какое-то интеллектуальное решение, которое будет отсекать плохие направлени€ на каждой итерации.



std::vector<Atom> Smoother::smooth(
	double length, 
	std::pair<int, int> endLine,
	std::pair<int, int> startLine,
	cv::Point2f start,
	cv::Point2f end,
	cv::Point2f startDirection
) {
	processor->prepare(endLine, startLine, start, end, startDirection);
	iterationProcess(processor->startAtom.point, processor->endAtom.point, length, *processor);
	return processor->getResult(); 
}