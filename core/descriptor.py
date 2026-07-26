"""
Дескриптор формы листа (профиль вращения) для Streamlit-приложения.

Обёртка над C++ модулем leaftools (папка leaf_cpp/). Инкапсулирует
подготовку данных: центр масс и площадь считаются по маске.

ВАЖНО про инверсию:
    Маски из segmentation.py — лист БЕЛЫЙ (255) на чёрном фоне.
    Поэтому инвертировать НЕ нужно (в отличие от .bmp сканов, где
    лист чёрный). Функции здесь принимают уже готовую маску лист=255.
"""

import numpy as np
import cv2


def _import_descriptor():
    """
    Импортирует C++ модуль. Вынесено в функцию, чтобы приложение
    не падало на старте, если модуль ещё не собран — ошибка появится
    только при реальном вызове дескриптора, с понятным сообщением.
    """
    try:
        import leaftools
        return leaftools
    except ImportError as e:
        raise ImportError(
            "Модуль leaftools не установлен. Соберите его из папки leaf_cpp/ "
            "(см. leaf_cpp/README.md) и установите wheel в текущий venv."
        ) from e


def centroid_and_area(mask: np.ndarray) -> tuple[int, int, int]:
    """
    Центр масс и площадь маски листа (лист = 255).

    Площадь — число ненулевых пикселей (countNonZero), а НЕ contourArea:
    дескриптору нужна именно попиксельная площадь для нормировки Жаккара.
    """
    if mask.ndim != 2:
        raise ValueError("Ожидается одноканальная маска (H, W)")
    M = cv2.moments(mask)
    if M["m00"] == 0:
        raise ValueError("Пустая маска: лист не найден")
    cx = round(M["m10"] / M["m00"])
    cy = round(M["m01"] / M["m00"])
    area = int(cv2.countNonZero(mask))
    return cx, cy, area


def describe(mask: np.ndarray) -> dict:
    """
    Считает дескриптор формы по маске листа (лист = 255, фон = 0).

    Возвращает dict:
        angles         : np.ndarray — углы 0…180
        jaccard_values : np.ndarray — мера Жаккара на каждом угле
        count          : int
        center         : (x, y)
        area           : int
    """
    leaftools = _import_descriptor()
    cx, cy, area = centroid_and_area(mask)
    res = leaftools.generate_descriptor(mask, cx, cy, area)
    res["center"] = (cx, cy)
    res["area"] = area
    return res


def descriptor_available() -> bool:
    """Проверка, собран ли C++ модуль (для мягкого поведения UI)."""
    try:
        import leaftools  # noqa: F401
        return True
    except ImportError:
        return False