"""
leaf_picker — Streamlit-компонент: выбор листа кликом по скану.

Поверх скана рисуются прозрачные зоны по контуру каждого листа.
При наведении лист заливается цветом (как кнопка под курсором), при клике
заливка остаётся — видно, какой лист выбран, — а индекс листа уходит в Python.

Пример:
    from components.leaf_picker import leaf_picker

    idx = leaf_picker(scan, [leaf["contour"] for leaf in leaves], key="scan1")
    if idx is not None:
        st.image(leaves[idx]["crop"])
"""

import base64
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import streamlit.components.v1 as components

# Папка с фронтендом (index.html лежит рядом с этим файлом).
_FRONTEND_DIR = Path(__file__).parent / "frontend"

# Регистрируем компонент один раз при импорте модуля.
_component_func = components.declare_component(
    "leaf_picker", path=str(_FRONTEND_DIR))

# Ширина, до которой ужимается скан перед отправкой в браузер.
# Сканы бывают по 25 МБ — гнать их целиком в iframe незачем.
_MAX_WIDTH = 900

# Допуск упрощения контура, в долях периметра.
# CHAIN_APPROX_NONE даёт точку на каждый пиксель границы — для листа это
# тысячи точек. Столько в браузере не нужно: упрощаем, форма сохраняется.
_SIMPLIFY_EPS = 0.0015


def _encode_scan(scan_bgr: np.ndarray, max_width: int) -> tuple[str, float]:
    """
    Ужимает скан по ширине и кодирует в base64 PNG.

    Возвращает (data-url, scale), где scale — во сколько раз уменьшено
    изображение: на него же нужно домножить координаты контуров.
    """
    h, w = scan_bgr.shape[:2]
    scale = min(max_width / w, 1.0)

    if scale < 1.0:
        new_size = (int(round(w * scale)), int(round(h * scale)))
        scan_bgr = cv2.resize(scan_bgr, new_size, interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(".png", scan_bgr)
    if not ok:
        raise ValueError("Не удалось закодировать скан в PNG")

    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}", scale


def _contour_to_points(contour: np.ndarray, scale: float,
                       eps_ratio: float) -> list[list[float]]:
    """
    Упрощает контур и переводит в список точек для SVG.

    contour   : (N, 1, 2) int32 — как отдаёт cv2.findContours;
    scale     : коэффициент уменьшения скана;
    eps_ratio : допуск approxPolyDP в долях периметра.
    """
    contour = np.asarray(contour, dtype=np.int32).reshape(-1, 1, 2)

    peri = cv2.arcLength(contour, True)
    if peri > 0 and eps_ratio > 0:
        contour = cv2.approxPolyDP(contour, eps_ratio * peri, True)

    pts = contour.reshape(-1, 2).astype(float) * scale
    # Округляем до десятых: точнее в SVG не нужно, а данных заметно меньше.
    return [[round(float(x), 1), round(float(y), 1)] for x, y in pts]


def leaf_picker(
        scan_bgr: np.ndarray,
        contours: Sequence[np.ndarray],
        labels: Sequence[str] | None = None,
        selected: int | None = None,
        max_width: int = _MAX_WIDTH,
        simplify: float = _SIMPLIFY_EPS,
        key: str | None = None,
):
    """
    Показывает скан с кликабельными зонами листьев.

    scan_bgr  : изображение скана (H, W, 3) uint8 BGR (как из cv2.imread).
    contours  : контуры листьев в координатах ИСХОДНОГО скана —
                поле "contour" из split_leaves.
    labels    : подписи зон для всплывающей подсказки (по умолчанию "Лист N").
    selected  : НАЧАЛЬНЫЙ выбор. Дальше подсветкой распоряжается сам
                компонент: он отмечает лист сразу по клику, не дожидаясь
                ответа от Python.
    max_width : до какой ширины ужимать скан перед показом.
    simplify  : допуск упрощения контура в долях периметра (0 — не упрощать).
    key       : ключ Streamlit. Обязателен, если компонентов на странице
                несколько, иначе они будут делить одно состояние.

    Возвращает индекс выбранного листа (int) или None, если ещё не кликали.
    """
    src, scale = _encode_scan(scan_bgr, max_width)
    h, w = scan_bgr.shape[:2]

    zones = [
        {
            "points": _contour_to_points(contour, scale, simplify),
            "label": labels[i] if labels is not None else f"Лист {i + 1}",
        }
        for i, contour in enumerate(contours)
    ]

    return _component_func(
        src=src,
        zones=zones,
        # Размеры уменьшенного скана — система координат для SVG.
        width=round(w * scale, 1),
        height=round(h * scale, 1),
        selected=selected,
        key=key,
        default=None,
    )
