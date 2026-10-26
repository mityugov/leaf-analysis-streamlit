import base64
from html import escape
from pathlib import Path

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from core import segmentation
from components.leaf_picker import leaf_picker

# Конфигурация страницы: заголовок вкладки браузера, иконка, широкий макет.
st.set_page_config(page_title="Анализ листьев", page_icon="🍃", layout="wide")

# Папка с моделью SVM для поиска черешка (лежит рядом с app.py).
MODELS_DIR = Path(__file__).parent / "models"

# Сторона квадрата, в который вписывается изображение листа в центральной
# колонке, пикселей. На странице квадрат растягивается на ширину колонки,
# поэтому высота изображения не зависит от формы листа.
LEAF_VIEW_SIZE = 800

# Изображения симметрии в порядке показа. Ось рисуется поверх текущего
# вида листа, остальные изображения возвращает поиск оси симметрии.
SYMMETRY_VIEWS = [
    ("axis", "Ось симметрии"),
    ("straightened", "Распрямлённый лист"),
    ("reflected", "Лист и его отражение"),
]


# Тяжёлые вычисления.
#   @st.cache_resource — модель, живёт всю сессию;
#   @st.cache_data     — результат по входу: тот же вход не считается заново.

@st.cache_resource(show_spinner="Обучение модели фон/лист…")
def get_svm():
    return segmentation.load_svm()


@st.cache_data(show_spinner=False)
def load_scan(image_bytes: bytes):
    """
    Декодирует файл скана в изображение BGR.

    Возвращает None, если файл не является изображением.
    """
    arr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


@st.cache_data(show_spinner=False)
def segment_scan_image(image_bytes: bytes):
    """
    Сегментация всего скана (SVM: фон/лист).

    Возвращает маску всего скана и список листьев: вырез, маска, рамка
    и контур каждого листа в координатах скана.
    """
    return segmentation.segment(load_scan(image_bytes), get_svm())


# Вспомогательные функции.

def leaf_name(file_name: str, leaf_idx: int) -> str:
    """Имя листа: "TCP13014.bmp" + индекс 0 -> "TCP13014_1.bmp"."""
    p = Path(file_name)
    return f"{p.stem}_{leaf_idx + 1}{p.suffix or '.bmp'}"


def apply_mask_to_color(crop_bgr, mask):
    """
    Оставляет на цветном изображении только то, что отмечено маской,
    остальное закрашивает белым.

    Используется для показа цветного листа без черешка: черешок стирается
    по маске пластины.
    """
    out = crop_bgr.copy()
    out[mask == 0] = (255, 255, 255)
    return out


def extend_to_edge(axis_points, mask):
    """
    Продлевает ось от последней точки до края листа по направлению
    последнего отрезка.

    Поиск оси идёт шагами и останавливается перед шагом, который вышел бы
    за лист, поэтому последняя точка лежит немного внутри листа.
    """
    if len(axis_points) < 2:
        return axis_points

    last = axis_points[-1]
    direction = last - axis_points[-2]
    length = np.hypot(direction[0], direction[1])
    if length == 0:
        return axis_points
    direction = direction / length

    h, w = mask.shape
    end = last
    while True:
        next_point = end + direction
        x = int(round(next_point[0]))
        y = int(round(next_point[1]))
        if x < 0 or y < 0 or x >= w or y >= h:
            break
        if mask[y, x] == 0:
            break
        end = next_point

    return np.vstack([axis_points, end])


def draw_axis(image, axis_points, mask):
    """
    Рисует ось симметрии красной линией поверх изображения листа.

    axis_points — точки оси (N, 2) в координатах изображения;
    mask        — маска, по которой искалась ось.

    Толщина линии зависит от размера изображения, чтобы после вписывания
    в квадрат LEAF_VIEW_SIZE линия была одинаково видна. Линия остаётся
    только внутри маски: у узкого кончика листа толстая линия не выходит
    за край.
    """
    if image.ndim == 2:
        out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        out = image.copy()

    h, w = out.shape[:2]
    thickness = max(1, round(1 * max(h, w) / LEAF_VIEW_SIZE))
    points = extend_to_edge(axis_points, mask)
    points = np.round(points).astype(np.int32).reshape(-1, 1, 2)

    with_line = out.copy()
    cv2.polylines(with_line, [points], False, (0, 0, 255), thickness, cv2.LINE_AA)
    inside = mask > 0
    out[inside] = with_line[inside]
    return out


def fit_to_box(image, box_h, box_w, background=(255, 255, 255)):
    """
    Вписывает изображение в рамку box_h x box_w с сохранением пропорций
    и полями цвета background по краям.

    Все изображения листа (цветное, маска, симметрия) вписываются в квадрат
    LEAF_VIEW_SIZE: длинный лист не уходит вниз, а при переключении видов
    изображение не меняет размер.

    image — изображение BGR или одноканальная маска; результат — BGR uint8.
    """
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    h, w = image.shape[:2]
    scale = min(box_h / h, box_w / w)
    new_w = max(int(round(w * scale)), 1)
    new_h = max(int(round(h * scale)), 1)

    # При уменьшении INTER_AREA сглаживает без ряби; при увеличении
    # INTER_NEAREST сохраняет чёткие края маски и линии оси.
    if scale < 1:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_NEAREST
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    canvas = np.full((box_h, box_w, 3), background, np.uint8)
    y0 = (box_h - new_h) // 2
    x0 = (box_w - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def show_leaf_image(image_bgr, caption: str):
    """
    Выводит изображение BGR uint8 с подписью.

    Изображение вставляется в страницу как PNG в data-URL. В отличие от
    st.image, браузеру не нужен отдельный запрос к серверу Streamlit:
    такой запрос мог прийти уже после того, как Streamlit удалил файл
    прошлого прогона, и тогда вместо листа показывалась «битая» картинка.
    """
    ok, buffer = cv2.imencode(".png", image_bgr)
    if not ok:
        st.error("Не удалось закодировать изображение листа")
        return

    b64 = base64.b64encode(buffer.tobytes()).decode("ascii")
    # Подпись экранируется: символы < и " в имени файла не ломают HTML.
    safe_caption = escape(caption)
    st.markdown(
        '<figure style="margin:0;width:100%;">'
        f'<img src="data:image/png;base64,{b64}" alt="{safe_caption}" '
        'style="display:block;width:100%;height:auto;object-fit:contain;'
        'border-radius:8px;"/>'
        f'<figcaption style="text-align:center;opacity:.75;'
        f'font-size:.9rem;margin-top:6px;">{safe_caption}</figcaption>'
        '</figure>',
        unsafe_allow_html=True,
    )


# Загрузка файлов.

st.title("🍃 Анализ листьев")

uploaded = st.file_uploader(
    "Загрузить сканированные изображения листьев",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
    accept_multiple_files=True,
)

if not uploaded:
    st.info("Выберите изображение(я) скана.")
    st.stop()

# При загрузке скан только декодируется. Листья находит сегментация,
# которая запускается кнопкой.
scans = {}       # имя файла -> изображение BGR
scan_bytes = {}  # имя файла -> байты файла (ключ кэша сегментации)
for f in uploaded:
    data = f.getvalue()
    image = load_scan(data)
    if image is not None:
        scans[f.name] = image
        scan_bytes[f.name] = data

if not scans:
    st.error("Не удалось прочитать ни одного изображения.")
    st.stop()

names = list(scans.keys())


# Состояние сессии.
#   sel_leaf — выбранный лист на каждом скане            {имя файла: индекс}
#   seg      — результаты сегментации сканов             {имя файла: res}
#   petiole  — результаты отрезания черешка              {(файл, индекс): res}
#   symmetry — результаты поиска оси симметрии           {(файл, индекс, без черенка): res}
#   sym_page — номер показанного изображения симметрии   {(файл, индекс): 0..2}
#   view_bin / view_nopet / view_sym — положение переключателей показа.
#
# Положение переключателей хранится в отдельных словарях: Streamlit стирает
# состояние виджетов, которых нет в текущей отрисовке, а на странице
# одновременно показан только один выбранный лист.
for name in ("sel_leaf", "seg", "petiole", "symmetry", "sym_page",
             "view_bin", "view_nopet", "view_sym"):
    if not isinstance(st.session_state.get(name), dict):
        st.session_state[name] = {}


def remember_toggle(store_name, store_key, widget_key):
    """
    Переносит положение переключателя в словарь store_name.

    Вызывается только при щелчке по чекбоксу. Значение самого виджета
    после переключения листьев приходит сброшенным, поэтому положение
    переключателя читается из словаря.
    """
    st.session_state[store_name][store_key] = st.session_state[widget_key]


def shift_sym_page(leaf_key, step):
    """Переключает изображение симметрии на соседнее (по кругу)."""
    page = st.session_state.sym_page.get(leaf_key, 0)
    st.session_state.sym_page[leaf_key] = (page + step) % len(SYMMETRY_VIEWS)


def segment_scan(fname):
    """Сегментирует скан целиком."""
    st.session_state.seg[fname] = segment_scan_image(scan_bytes[fname])


# Верхняя панель: сегментация всех сканов сразу и общий прогресс.

not_segmented = [fname for fname in names if fname not in st.session_state.seg]
n_done = len(names) - len(not_segmented)

top_l, top_r = st.columns([1, 3])
with top_l:
    if st.button("▶ Сегментировать всё", type="primary",
                 width="stretch", disabled=not not_segmented):
        bar = st.progress(0.0, text="Сегментация…")
        for done, fname in enumerate(not_segmented):
            bar.progress(done / len(not_segmented),
                         text=f"Сегментация: {fname}")
            segment_scan(fname)
        bar.empty()
        st.rerun()
with top_r:
    st.progress(n_done / len(names),
                text=f"Сегментировано сканов: {n_done} из {len(names)}")

st.markdown("---")


# Вкладки по сканам.
#
# Streamlit выполняет код всех вкладок сразу, а браузер прячет неактивные.
# Поэтому здесь нет тяжёлых вычислений: сегментация, черешок, симметрия
# и дескриптор запускаются по кнопке, а результат хранится в session_state.
#
# Названия вкладок не меняются после сегментации: при смене названий
# Streamlit переключил бы страницу на первую вкладку.
tabs = st.tabs(names)

for tab, fname in zip(tabs, names):
    with tab:
        seg_res = st.session_state.seg.get(fname)
        leaves = seg_res["leaves"] if seg_res is not None else []
        k = len(leaves)

        col_scan, col_leaf, col_ctl = st.columns([2, 2, 1.3])

        # Слева: скан с подсветкой листьев. До сегментации контуров нет,
        # и выбирать на скане нечего.
        with col_scan:
            clicked = leaf_picker(
                scan_bgr=scans[fname],
                contours=[leaf["contour"] for leaf in leaves],
                labels=[leaf_name(fname, i) for i in range(k)],
                selected=st.session_state.sel_leaf.get(fname),
                key=f"picker_{fname}",
            )
            if seg_res is None:
                st.caption("Сегментируйте скан — после этого листья можно "
                           "будет выбирать кликом.")
            else:
                st.caption("Кликните по листу, чтобы открыть его справа.")

        # Индекс приходит только по клику и проверяется: он пришёл из браузера.
        if clicked is not None:
            clicked = int(clicked)
            if 0 <= clicked < k:
                st.session_state.sel_leaf[fname] = clicked

        idx = st.session_state.sel_leaf.get(fname)

        # Справа: кнопки и переключатели. Каждой операции соответствуют
        # кнопка (посчитать) и чекбокс (показать); чекбокс заблокирован,
        # пока показывать нечего.
        with col_ctl:
            # Сегментация относится ко всему скану, а не к выбранному листу.
            # Результат кэшируется по содержимому файла, поэтому повторная
            # сегментация того же скана не нужна.
            if seg_res is None:
                if st.button("▶ Сегментировать скан", key=f"seg_{fname}",
                             width="stretch", type="primary"):
                    with st.spinner("Сегментация…"):
                        segment_scan(fname)
                    st.rerun()
            else:
                st.button(f"✔ Сегментировано, листьев: {k}",
                          key=f"seg_{fname}", width="stretch", disabled=True)

        # Дальше всё привязано к выбранному листу.
        if seg_res is None:
            with col_leaf:
                st.info("Скан ещё не сегментирован.")
            continue

        if k == 0:
            with col_leaf:
                st.warning("Листья не найдены. Проверьте контраст с фоном.")
            continue

        if idx is None or not (0 <= idx < k):
            with col_leaf:
                st.info("Лист не выбран — кликните по листу на скане.")
            continue

        leaf = leaves[idx]
        cut_res = st.session_state.petiole.get((fname, idx))

        with col_ctl:
            st.markdown(f"**{leaf_name(fname, idx)}**")

            # Бинарный вид — способ смотреть на скан, поэтому положение
            # переключателя общее для всех его листьев (ключ в словаре —
            # имя файла).
            bin_key = f"bin_{fname}_{idx}"
            st.checkbox(
                "Бинарный вид",
                value=st.session_state.view_bin.get(fname, False),
                key=bin_key,
                on_change=remember_toggle,
                args=("view_bin", fname, bin_key),
            )
            show_binary = st.session_state.view_bin.get(fname, False)

            st.markdown("")

            # Отрезание черешка разблокирует показ без черешка.
            if st.button("✂ Удалить черенок", key=f"cutbtn_{fname}_{idx}",
                         width="stretch"):
                try:
                    import leaftools
                    with st.spinner("Поиск черенка…"):
                        st.session_state.petiole[(fname, idx)] = \
                            leaftools.cut_petiole(
                                leaf["mask"],
                                svm_model_path=str(MODELS_DIR / "svm_model.xml"),
                                svm_csv_path=str(MODELS_DIR / "SVM_train_7_1.csv"))
                    # Ось, найденная по прежней пластине, больше не подходит.
                    st.session_state.symmetry.pop((fname, idx, True), None)
                    st.rerun()
                except ImportError:
                    st.error("Модуль leaftools не установлен.")
                except Exception as e:
                    st.error(f"Не удалось отрезать черенок: {e}")

            petiole_ready = cut_res is not None and cut_res.get("found")
            # Черешок отрезается для каждого листа отдельно, поэтому
            # и положение переключателя хранится для каждого листа.
            nopet_key = f"nopet_{fname}_{idx}"
            st.checkbox(
                "Без черенка",
                value=st.session_state.view_nopet.get((fname, idx), False),
                key=nopet_key,
                disabled=not petiole_ready,
                help=None if petiole_ready else "Сначала удалите черенок",
                on_change=remember_toggle,
                args=("view_nopet", (fname, idx), nopet_key),
            )
            hide_petiole = st.session_state.view_nopet.get((fname, idx), False)

            if cut_res is not None and not cut_res.get("found"):
                st.caption("⚠ Точки черенка не найдены.")

            # Маска, по которой считаются признаки: пластина или весь лист.
            use_blade = hide_petiole and petiole_ready
            active_mask = leaf["mask"]
            if use_blade:
                active_mask = cut_res["blade"]

            # Ось симметрии листа с черенком и без него хранится отдельно.
            # Показывается та, что соответствует переключателю «Без черенка».
            sym_id = (fname, idx, use_blade)
            sym_res = st.session_state.symmetry.get(sym_id)
            if use_blade:
                mask_note = "без черенка"
            else:
                mask_note = "с черенком"

            st.markdown("")

            # Ось симметрии считается по той маске, что показана.
            if st.button("⟷ Ось симметрии", key=f"symbtn_{fname}_{idx}",
                         width="stretch"):
                try:
                    import leaftools
                    with st.spinner("Поиск оси симметрии…"):
                        st.session_state.symmetry[sym_id] = \
                            leaftools.symmetry(active_mask)
                    st.session_state.sym_page[(fname, idx)] = 0
                    st.rerun()
                except ImportError:
                    st.error("Модуль leaftools не установлен.")
                except Exception as e:
                    st.error(f"Не удалось найти ось симметрии: {e}")

            symmetry_ready = sym_res is not None and sym_res.get("found")
            sym_key = f"sym_{fname}_{idx}"
            st.checkbox(
                "Симметрия",
                value=st.session_state.view_sym.get((fname, idx), False),
                key=sym_key,
                disabled=not symmetry_ready,
                help=None if symmetry_ready else
                f"Сначала найдите ось симметрии для листа {mask_note}",
                on_change=remember_toggle,
                args=("view_sym", (fname, idx), sym_key),
            )
            show_symmetry = st.session_state.view_sym.get((fname, idx), False)

            if sym_res is not None and not sym_res.get("found"):
                st.caption("⚠ Ось симметрии не найдена.")

            st.markdown("")

            # Профиль вращения считается по той маске, что показана.
            want_profile = st.button(
                "📈 Профиль вращения", key=f"prof_{fname}_{idx}",
                width="stretch")

        # В центре: выбранный лист в нужном виде. Показ зависит
        # от переключателей:
        #   бинарный вид — маска вместо цветного изображения;
        #   без черенка  — вместо полной маски берётся пластина, а на
        #                  цветном изображении черешок стирается по ней;
        #   симметрия    — изображения симметрии со стрелками переключения.
        #                  Ось рисуется поверх текущего вида листа.
        with col_leaf:
            if show_binary:
                leaf_view = active_mask
                # У маски фон чёрный, поэтому и поля чёрные.
                background = (0, 0, 0)
                caption = leaf_name(fname, idx) + " · маска"
            else:
                leaf_view = leaf["crop"]
                if use_blade:
                    leaf_view = apply_mask_to_color(leaf_view, cut_res["blade"])
                background = (255, 255, 255)
                caption = leaf_name(fname, idx)

            if show_symmetry and symmetry_ready:
                page = st.session_state.sym_page.get((fname, idx), 0)
                view_key, view_title = SYMMETRY_VIEWS[page]
                if view_key == "axis":
                    image = draw_axis(leaf_view, sym_res["axis_points"],
                                      active_mask)
                else:
                    image = sym_res[view_key]
                    background = (255, 255, 255)
                show_leaf_image(
                    fit_to_box(image, LEAF_VIEW_SIZE, LEAF_VIEW_SIZE,
                               background=background),
                    caption=f"{view_title} · {page + 1}/{len(SYMMETRY_VIEWS)}"
                            f" · {mask_note}",
                )

                prev_col, info_col, next_col = st.columns([1, 3, 1])
                with prev_col:
                    st.button("◀", key=f"symprev_{fname}_{idx}",
                              width="stretch",
                              on_click=shift_sym_page,
                              args=((fname, idx), -1))
                with info_col:
                    st.markdown(
                        f"Жаккар: **{sym_res['jaccard_straightened']:.3f}** "
                        f"(по исходной площади "
                        f"{sym_res['jaccard_original']:.3f})")
                with next_col:
                    st.button("▶", key=f"symnext_{fname}_{idx}",
                              width="stretch",
                              on_click=shift_sym_page,
                              args=((fname, idx), 1))
            else:
                show_leaf_image(
                    fit_to_box(leaf_view, LEAF_VIEW_SIZE, LEAF_VIEW_SIZE,
                               background=background),
                    caption=caption)

        # Под колонками: профиль вращения. Результат не кэшируется:
        # дескриптор считается быстро, а маска зависит от переключателей.
        if want_profile:
            try:
                import leaftools
                desc = leaftools.describe_mask(active_mask)

                chart_df = pd.DataFrame({
                    "angle": desc["angles"],
                    "jaccard": desc["jaccard_values"],
                })
                chart = (
                    alt.Chart(chart_df)
                    .mark_line()
                    .encode(
                        x=alt.X("angle:Q", scale=alt.Scale(domain=[0, 180]),
                                title="Угол"),
                        y=alt.Y("jaccard:Q", scale=alt.Scale(domain=[0, 1]),
                                title="Жаккар"),
                    )
                    .properties(height=220)
                )
                st.altair_chart(chart, width="stretch")

                csv = "angle;jaccard\n" + "\n".join(
                    f"{int(a)};{v:.6f}"
                    for a, v in zip(desc["angles"], desc["jaccard_values"]))
                st.download_button(
                    "Скачать дескриптор (CSV)", data=csv,
                    file_name=f"{leaf_name(fname, idx)}_descriptor.csv",
                    mime="text/csv", key=f"csv_{fname}_{idx}")
            except ImportError:
                st.error("Модуль leaftools не установлен.")
            except Exception as e:
                st.error(f"Не удалось посчитать дескриптор: {e}")