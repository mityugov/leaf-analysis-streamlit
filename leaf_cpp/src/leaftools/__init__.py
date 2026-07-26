"""
leaftools — дескриптор формы листа (профиль вращения) на C++/OpenCV.

Низкоуровневая функция из C++:
    generate_descriptor(mask, cx, cy, area) -> dict

Удобная обёртка (готовит данные по стандартной схеме):
    describe_leaf(path_or_array, invert=True) -> dict
"""

from ._core import generate_descriptor, cv_version

__all__ = ["generate_descriptor", "cv_version", "describe_leaf", "describe_mask"]


def _centroid_and_area(mask):
    import cv2
    M = cv2.moments(mask)
    if M["m00"] == 0:
        raise ValueError("Пустая маска: объект не найден (m00 = 0)")
    cx = round(M["m10"] / M["m00"])
    cy = round(M["m01"] / M["m00"])
    area = int(cv2.countNonZero(mask))
    return cx, cy, area


def describe_mask(mask):
    """
    Дескриптор для ГОТОВОЙ маски (лист = 255, фон = 0).
    Возвращает dict: angles, jaccard_values, count, center, area.
    """
    cx, cy, area = _centroid_and_area(mask)
    res = generate_descriptor(mask, cx, cy, area)
    res["center"] = (cx, cy)
    res["area"] = area
    return res


def describe_leaf(source, invert=True):
    """
    Дескриптор формы листа из файла или numpy-массива.

    source : путь к изображению (str) или numpy (H, W) uint8.
    invert : True  — лист ЧЁРНЫЙ на белом фоне (как .bmp сканы), инвертируем.
             False — лист УЖЕ белый на чёрном (маска из сегментации).

    Возвращает dict: angles, jaccard_values, count, center, area.
    """
    import cv2
    import numpy as np

    if isinstance(source, str):
        image = cv2.imread(source, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Не удалось прочитать изображение: {source}")
    else:
        image = source
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mask = (255 - image) if invert else image
    return describe_mask(mask)