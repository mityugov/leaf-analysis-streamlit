"""
leaftools — дескриптор формы листа (профиль вращения) на C++/OpenCV.

Низкоуровневая функция из C++:
    generate_descriptor(mask, cx, cy, area) -> dict

Удобная обёртка (готовит данные по стандартной схеме):
    describe_leaf(path_or_array, invert=True) -> dict
"""

from ._core import generate_descriptor, cv_version, find_petiole_points

__all__ = ["generate_descriptor", "cv_version", "describe_leaf", "describe_mask",
           "find_petiole_points", "cut_petiole", "draw_petiole_points"]


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


# ─── Отрезание черешка (гибрид: точки из C++, разрезание на Python) ──────────

def _nearest_index_on_contour(pt, contour):
    """Индекс ближайшей к pt точки контура. contour: (N,1,2) от cv2.findContours."""
    import numpy as np
    pts = contour.reshape(-1, 2)
    d = np.sum((pts - np.array(pt)) ** 2, axis=1)
    return int(np.argmin(d))


def _arc(pts, start, end):
    """Дуга контура от start до end включительно, идя вперёд по индексам
    с переходом через конец (после n-1 снова 0)."""
    n = len(pts)
    out = [pts[start]]
    i = start
    while i != end:
        i = (i + 1) % n
        out.append(pts[i])
    return out


def _split_contour_by_three(contour, idx_a, idx_b, idx_s):
    """
    Разрезает контур на два сегмента по трём точкам A, B, S.

    Черешок — это дуга, идущая от A через S к B (S лежит на основании
    черешка между A и B). Контур замкнут, поэтому дуга может проходить
    через "конец" контура (после последнего индекса — снова 0-й) — это
    обрабатывается движением по модулю n.

    Возвращает:
      segment_with_points — дуга A→S→B (ЧЕРЕШОК),
      other_segment       — оставшаяся дуга B→...→A (ПЛАСТИНА).
    """
    import numpy as np
    pts = contour.reshape(-1, 2)
    n = len(pts)

    # Есть два пути от A к B по замкнутому контуру: вперёд и назад.
    # Черенок — тот путь, который проходит ЧЕРЕЗ точку S.
    # Проверяем, лежит ли S на дуге A→B (идя вперёд от A к B).
    def arc_contains(start, end, target):
        i = start
        while i != end:
            if i == target:
                return True
            i = (i + 1) % n
        return i == target  # проверить и конечную

    if arc_contains(idx_a, idx_b, idx_s):
        # S на дуге A→B (вперёд) — это черешок
        segment_with_points = _arc(pts, idx_a, idx_b)
        other_segment = _arc(pts, idx_b, idx_a)
    else:
        # S на дуге B→A (вперёд) — черешок идёт от B к A
        segment_with_points = _arc(pts, idx_b, idx_a)
        other_segment = _arc(pts, idx_a, idx_b)

    seg = np.array(segment_with_points, dtype=np.int32) if segment_with_points else None
    oth = np.array(other_segment, dtype=np.int32) if other_segment else None
    return seg, oth


def cut_petiole(mask, svm_model_path="", svm_csv_path=""):
    """
    Отделяет листовую пластину от черешка.

    mask : бинарная маска (H,W) uint8, лист = 255 на чёрном фоне.
    svm_model_path : путь к обученной модели SVM (опционально).
    svm_csv_path   : путь к CSV для обучения SVM, если модели нет (опционально).

    Возвращает dict:
        found   : bool — найдены ли точки черешка;
        blade   : маска пластины (255 = пластина) той же формы, что mask;
        petiole : маска черешка (255 = черешок);
        points  : {'A','B','S': (x,y)} — точки отреза НА КОНТУРЕ
                  (спроецированные на ближайшую точку контура);
        indices : {'A','B','S': int} — индексы этих точек в контуре findContours.

    Если точки не найдены (found=False), blade = исходная маска, petiole пустой,
    поля points/indices отсутствуют.
    """
    import cv2
    import numpy as np

    # 1. Точки черешка из C++.
    pts = find_petiole_points(mask, svm_model_path, svm_csv_path)
    found = bool(pts["found"])

    blade = mask.copy()
    petiole = np.zeros_like(mask)

    result = {
        "found": found,
        "blade": blade,
        "petiole": petiole,
    }

    if not found:
        return result

    # 2. Наибольший контур листа.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return result
    contour = max(contours, key=cv2.contourArea)

    # 3. Индексы ближайших точек контура к A, B, S (сырые точки из алгоритма).
    idx_a = _nearest_index_on_contour((pts["ax"], pts["ay"]), contour)
    idx_b = _nearest_index_on_contour((pts["bx"], pts["by"]), contour)
    idx_s = _nearest_index_on_contour((pts["sx"], pts["sy"]), contour)

    # Точки на контуре (спроецированные) и их индексы — это то, что реально
    # используется для разреза.
    pts_arr = contour.reshape(-1, 2)
    result["indices"] = {"A": idx_a, "B": idx_b, "S": idx_s}
    result["points"] = {
        "A": tuple(int(v) for v in pts_arr[idx_a]),
        "B": tuple(int(v) for v in pts_arr[idx_b]),
        "S": tuple(int(v) for v in pts_arr[idx_s]),
    }

    # 4. Разрез контура: сегмент с точками = черешок, остальное = пластина.
    seg_petiole, seg_blade = _split_contour_by_three(contour, idx_a, idx_b, idx_s)

    # 5. Заливка сегментов в маски.
    blade = np.zeros_like(mask)
    petiole = np.zeros_like(mask)
    if seg_blade is not None and len(seg_blade) > 0:
        cv2.drawContours(blade, [seg_blade], -1, 255, cv2.FILLED)
    if seg_petiole is not None and len(seg_petiole) > 0:
        cv2.drawContours(petiole, [seg_petiole], -1, 255, cv2.FILLED)

    result["blade"] = blade
    result["petiole"] = petiole
    return result


def draw_petiole_points(mask, cut_result, radius=6):
    """
    Рисует точки отреза на цветном изображении маски (для визуальной проверки).

    mask       : исходная маска (H,W) uint8.
    cut_result : результат cut_petiole (нужно поле points).
    radius     : радиус кружков.

    Возвращает BGR-изображение (H,W,3): A и B розовые, S красная.
    Если точки не найдены — просто цветная копия маски.
    """
    import cv2

    vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    if not cut_result.get("found"):
        return vis

    poc = cut_result.get("points")
    if not poc:
        return vis

    cv2.circle(vis, poc["A"], radius, (255, 0, 255), cv2.FILLED)  # розовый
    cv2.circle(vis, poc["B"], radius, (255, 0, 255), cv2.FILLED)  # розовый
    cv2.circle(vis, poc["S"], radius, (0, 0, 255), cv2.FILLED)    # красный
    return vis