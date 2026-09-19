#ifndef ROTATION_PROFILE_H
#define ROTATION_PROFILE_H

#include <opencv2/opencv.hpp>
#include <map>

/**
 * @brief Строит дескриптор формы - профиль вращения.
 *
 * Фигура последовательно поворачивается вокруг заданного центра на углы
 * 0…180° с шагом 1°, и для каждого угла вычисляется мера Жаккара между
 * повёрнутой и исходной фигурой.
 *
 * @param src    Бинарная маска фигуры (CV_8UC1, объект = ненулевые пиксели).
 * @param center Центр вращения (обычно центр масс фигуры).
 * @param area   Площадь фигуры в пикселях (для подсчета меры Жаккара).
 *
 * @return Карта «угол (0…180) → мера Жаккара [0..1]», 181 элемент.
 */
std::map<double, double> generateDescriptorSequence(
        cv::Mat src,
        cv::Point center,
        int area);

#endif // ROTATION_PROFILE_H
