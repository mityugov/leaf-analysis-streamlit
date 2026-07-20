"""
Нарезка скана на отдельные листья.
"""

import cv2
import numpy as np

MIN_AREA = 2000     # минимальная площадь контура (отсекает мусор и пыль)
MAX_ASPECT = 50     # максимальная вытянутость (отсекает линейки, царапины)
PADDING = 25        # отступ вокруг листа при вырезании, пикселей


def binarize_otsu(image_bgr: np.ndarray) -> np.ndarray:
    """
    Бинаризует изображение методом Оцу.

    Порог подбирается автоматически по гистограмме яркости. Считается, что
    лист темнее фона (тёмный лист на светлом скане), поэтому используется
    THRESH_BINARY_INV — после инверсии лист становится белым.

    Параметры
    ---------
    image_bgr : np.ndarray
        Цветное изображение (H, W, 3), dtype uint8, порядок каналов BGR
        (как возвращает cv2.imread / cv2.imdecode).

    Возвращает
    ----------
    np.ndarray
        Бинарная маска (H, W), dtype uint8: лист = 255, фон = 0.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def find_leaf_contours(
        binary: np.ndarray,
        min_area: float = MIN_AREA,
        max_aspect: float = MAX_ASPECT,
) -> list[np.ndarray]:
    """
    Ищет контуры листьев на бинарной маске и отбрасывает мусор.

    Берутся только внешние контуры (дырки и прожилки внутри листа не нужны).
    Каждый контур проверяется по двум критериям: площадь (отсекает пыль и
    мелкие пятна) и вытянутость bounding box (отсекает линейки, царапины,
    полосы по краю скана).

    Параметры
    ---------
    binary : np.ndarray
        Бинарная маска (H, W), dtype uint8, где объект = 255.
    min_area : float
        Минимальная площадь контура в пикселях. Контуры меньше отбрасываются.
    max_aspect : float
        Максимально допустимое отношение сторон bounding box (длинная/короткая).
        Контуры более вытянутые отбрасываются.

    Возвращает
    ----------
    list[np.ndarray]
        Список контуров-кандидатов. Каждый контур — массив точек формы
        (N, 1, 2), dtype int32 (формат OpenCV). Пустой список, если ничего
        не прошло фильтр.
    """
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    good = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w == 0 or h == 0:
            continue
        if max(w / h, h / w) > max_aspect:
            continue

        good.append(contour)

    return good


def crop_by_bbox(
        image_bgr: np.ndarray,
        contour: np.ndarray,
        padding: int = PADDING,
        square: bool = False,
        mask_background: bool = False,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Вырезает один лист из изображения по ограничивающему прямоугольнику контура.

    К bounding box контура добавляется отступ padding со всех сторон
    (обрезается по границам изображения). Опционально прямоугольник
    расширяется до квадрата, а фон вокруг листа заливается белым по маске.

    Параметры
    ---------
    image_bgr : np.ndarray
        Исходное изображение (H, W, 3), dtype uint8, BGR.
    contour : np.ndarray
        Контур листа в формате OpenCV (N, 1, 2), dtype int32.
    padding : int
        Отступ вокруг листа в пикселях.
    square : bool
        Если True, bounding box расширяется до квадрата (по большей стороне,
        от центра) — удобно для подачи в нейросети с квадратным входом.
    mask_background : bool
        Если True, фон вокруг листа заливается белым по маске контура.
        Маска предварительно раздувается (dilate) на padding пикселей,
        чтобы белый фон не срезал тонкие края и зубчики листа.

    Возвращает
    ----------
    tuple[np.ndarray, tuple[int, int, int, int]]
        crop : np.ndarray
            Вырезанный фрагмент (h, w, 3), dtype uint8, BGR.
        bbox : tuple[int, int, int, int]
            Координаты выреза на исходном изображении: (x, y, w, h).
    """
    H, W = image_bgr.shape[:2]
    x, y, w, h = cv2.boundingRect(contour)

    if square:
        # расширяем меньшую сторону до размера большей, от центра
        side = max(w, h)
        cx = x + w // 2
        cy = y + h // 2
        x = cx - side // 2
        y = cy - side // 2
        w = h = side

    # применяем отступ и обрезаем по границам изображения
    x0 = max(x - padding, 0)
    y0 = max(y - padding, 0)
    x1 = min(x + w + padding, W)
    y1 = min(y + h + padding, H)

    crop = image_bgr[y0:y1, x0:x1].copy()

    if mask_background:
        leaf_mask = np.zeros((H, W), np.uint8)
        cv2.drawContours(leaf_mask, [contour], -1, 255, thickness=-1)

        # раздуваем маску, чтобы белый фон не срезал края листа
        grow = max(padding, 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow))
        leaf_mask = cv2.dilate(leaf_mask, kernel)

        crop_mask = leaf_mask[y0:y1, x0:x1]
        crop[crop_mask == 0] = (255, 255, 255)

    return crop, (x0, y0, x1 - x0, y1 - y0)


def split_leaves(
        image_bgr: np.ndarray,
        min_area: float = MIN_AREA,
        max_aspect: float = MAX_ASPECT,
        padding: int = PADDING,
        square: bool = False,
        mask_background: bool = False,
) -> list[dict[str, Any]]:
    """
    Делит скан на отдельные листья: бинаризация → контуры → вырезы.

    Объединяет весь конвейер нарезки. Для каждого найденного листа возвращает
    его вырез и геометрические признаки контура.

    Параметры
    ---------
    image_bgr : np.ndarray
        Изображение скана (H, W, 3), dtype uint8, BGR (как из cv2.imread).
    min_area : float
        Минимальная площадь контура (см. find_leaf_contours).
    max_aspect : float
        Максимальная вытянутость контура (см. find_leaf_contours).
    padding : int
        Отступ вокруг листа при вырезании (см. crop_by_bbox).
    square : bool
        Расширять ли вырез до квадрата (см. crop_by_bbox).
    mask_background : bool
        Заливать ли фон вокруг листа белым (см. crop_by_bbox).

    Возвращает
    ----------
    list[dict[str, Any]]
        Список листьев, по одному словарю на лист, со следующими ключами:
            crop        : np.ndarray — вырезанный лист (h, w, 3) uint8 BGR
            bbox        : tuple[int, int, int, int] — (x, y, w, h) на скане
            area        : float — площадь контура в пикселях
            perimeter   : float — длина контура в пикселях
            form_factor : float — 4·π·area / perimeter² (компактность, ~1 у круга)
        Пустой список, если листья не найдены.
    """
    binary = binarize_otsu(image_bgr)
    contours = find_leaf_contours(binary, min_area, max_aspect)

    leaves: list[dict[str, Any]] = []
    for contour in contours:
        crop, bbox = crop_by_bbox(
            image_bgr, contour, padding, square, mask_background)

        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        form_factor = 4 * np.pi * area / perimeter ** 2 if perimeter > 0 else 0.0

        leaves.append({
            "crop": crop,
            "bbox": bbox,
            "area": float(area),
            "perimeter": float(perimeter),
            "form_factor": float(form_factor),
        })

    return leaves