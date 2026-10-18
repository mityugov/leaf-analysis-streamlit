import numpy as np
import cv2
import pandas as pd
import streamlit as st
import altair as alt
from pathlib import Path

from core import segmentation
from core.leaf_splitter import split_leaf_scan_into_parts
from components.leaf_picker import leaf_picker

# Конфигурация страницы: заголовок вкладки браузера, иконка, широкий макет.
st.set_page_config(page_title="Анализ листьев", page_icon="🍃", layout="wide")

# Папка с моделью SVM для поиска черешка (лежит рядом с app.py).
MODELS_DIR = Path(__file__).parent / "models"


# ═══════════════════════════════════════════════════════════════════════════
#  Тяжёлые вычисления
#  @st.cache_resource — модель, живёт всю сессию.
#  @st.cache_data — результат по входу: тот же вход не считается заново.
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Обучение модели фон/лист…")
def get_svm():
    return segmentation.load_svm()


@st.cache_data(show_spinner=False)
def cut_leaves(image_bytes: bytes):
    """Декодирует скан и нарезает его на отдельные листья."""
    arr = np.frombuffer(image_bytes, np.uint8)
    image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return None

    leaves = split_leaf_scan_into_parts(image_bgr, mask_background=True)
    return {"scan": image_bgr, "leaves": leaves}


@st.cache_data(show_spinner=False)
def segment_leaf(leaf_bytes: bytes):
    """Сегментация одного листа (SVM: фон/лист)."""
    arr = np.frombuffer(leaf_bytes, np.uint8)
    crop = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return segmentation.segment(crop, get_svm())


# ═══════════════════════════════════════════════════════════════════════════
#  Вспомогательные функции
# ═══════════════════════════════════════════════════════════════════════════

def encode(img):
    """Кодирует изображение в PNG-байты (для кэшируемой функции)."""
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def rgb(img):
    """OpenCV хранит BGR, Streamlit ждёт RGB — конвертируем перед показом."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def leaf_name(file_name: str, leaf_idx: int) -> str:
    """Имя листа: "TCP13014.bmp" + индекс 0 -> "TCP13014_1.bmp"."""
    p = Path(file_name)
    return f"{p.stem}_{leaf_idx + 1}{p.suffix or '.bmp'}"


def leaf_zones(leaves):
    """
    Контуры листьев для компонента.

    Берём поле "contour" из сплиттера; если его нет (старая версия
    leaf_splitter.py) — строим прямоугольник по bbox, чтобы страница
    всё равно работала.
    """
    zones = []
    for leaf in leaves:
        contour = leaf.get("contour")
        if contour is not None:
            zones.append(contour)
        else:
            x, y, w, h = leaf["bbox"]
            zones.append(np.array(
                [[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.int32))
    return zones


def apply_mask_to_color(crop_bgr, mask):
    """
    Оставляет на цветном изображении только то, что отмечено маской,
    остальное закрашивает белым.

    Нужно для показа цветного листа без черешка: на цветной картинке
    черешок никуда не девается сам — его надо стереть по маске пластины.
    """
    out = crop_bgr.copy()
    out[mask == 0] = (255, 255, 255)
    return out


def fit_to_box(img_bgr, box_h, box_w):
    """
    Вписывает изображение в рамку box_h x box_w с сохранением пропорций
    и белыми полями по краям.

    Изображения симметрии обрезаны по рамке листа и по размеру не совпадают
    с изображением листа; так они занимают на странице то же место.
    """
    h, w = img_bgr.shape[:2]
    scale = min(box_h / h, box_w / w)
    new_w = max(int(round(w * scale)), 1)
    new_h = max(int(round(h * scale)), 1)
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    canvas = np.full((box_h, box_w, 3), 255, np.uint8)
    y0 = (box_h - new_h) // 2
    x0 = (box_w - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


# Изображения, которые возвращает поиск оси симметрии, в порядке показа.
SYMMETRY_VIEWS = [
    ("axis", "Ось симметрии"),
    ("straightened", "Распрямлённый лист"),
    ("reflected", "Лист и его отражение"),
]


# ═══════════════════════════════════════════════════════════════════════════
#  ЗАГРУЗКА ФАЙЛОВ
# ═══════════════════════════════════════════════════════════════════════════

st.title("🍃 Анализ листьев")

uploaded = st.file_uploader(
    "Загрузить сканированные изображения листьев",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
    accept_multiple_files=True,
)

if not uploaded:
    st.info("Выберите изображение(я) — листья нарежутся автоматически.")
    st.stop()

cuts = {}
for f in uploaded:
    res = cut_leaves(f.getvalue())
    if res is not None:
        cuts[f.name] = res

if not cuts:
    st.error("Не удалось прочитать ни одного изображения.")
    st.stop()

names = list(cuts.keys())
total_leaves = sum(len(c["leaves"]) for c in cuts.values())


# ═══════════════════════════════════════════════════════════════════════════
#  СОСТОЯНИЕ СЕССИИ
#    sel_leaf — выбранный лист на каждом скане        {имя файла: индекс}
#    seg      — результаты сегментации                {(файл, индекс): res}
#    petiole  — результаты отрезания черешка          {(файл, индекс): res}
#    symmetry — результаты поиска оси симметрии       {(файл, индекс): res}
#    sym_page — номер показанного изображения симметрии {(файл, индекс): 0..2}
# ═══════════════════════════════════════════════════════════════════════════

#    view_bin / view_nopet / view_sym — положение переключателей показа.
#
#    Хранить их приходится самим: Streamlit стирает состояние виджетов,
#    которых нет в текущей отрисовке, а на странице одновременно живёт
#    только один выбранный лист. Ключа виджета для этого мало.
for name in ("sel_leaf", "seg", "petiole", "symmetry", "sym_page",
             "view_bin", "view_nopet", "view_sym"):
    if not isinstance(st.session_state.get(name), dict):
        st.session_state[name] = {}


def remember_toggle(store_name, store_key, widget_key):
    """
    Переносит положение переключателя в наше хранилище.

    Вызывается только когда пользователь действительно щёлкнул по
    чекбоксу. Возвращаемое значение самого виджета мы не используем:
    Streamlit стирает состояние виджетов, исчезнувших с экрана, и после
    переключения листьев оно приходит сброшенным. Словарь же Streamlit
    не трогает — он и остаётся источником правды.
    """
    st.session_state[store_name][store_key] = st.session_state[widget_key]


def shift_sym_page(leaf_key, step):
    """Переключает изображение симметрии на соседнее (по кругу)."""
    page = st.session_state.sym_page.get(leaf_key, 0)
    st.session_state.sym_page[leaf_key] = (page + step) % len(SYMMETRY_VIEWS)


def segment_scan(fname):
    """Сегментирует все листья одного скана."""
    for i, leaf in enumerate(cuts[fname]["leaves"]):
        st.session_state.seg[(fname, i)] = segment_leaf(encode(leaf["crop"]))


# ═══════════════════════════════════════════════════════════════════════════
#  ВЕРХНЯЯ ПАНЕЛЬ: сегментация всех сканов сразу + общий прогресс
# ═══════════════════════════════════════════════════════════════════════════

n_done = len(st.session_state.seg)

top_l, top_r = st.columns([1, 3])
with top_l:
    if st.button("▶ Сегментировать всё", type="primary",
                 width="stretch"):
        bar = st.progress(0.0, text="Сегментация…")
        done = 0
        for fname in names:
            for i, leaf in enumerate(cuts[fname]["leaves"]):
                bar.progress(done / max(total_leaves, 1),
                             text=f"Сегментация: {leaf_name(fname, i)}")
                st.session_state.seg[(fname, i)] = segment_leaf(
                    encode(leaf["crop"]))
                done += 1
        bar.empty()
        st.rerun()
with top_r:
    st.progress(n_done / max(total_leaves, 1),
                text=f"Сегментировано листьев: {n_done} из {total_leaves}")

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════
#  ВКЛАДКИ ПО СКАНАМ
#
#  Streamlit выполняет код ВСЕХ вкладок сразу, а браузер прячет неактивные.
#  Поэтому здесь не должно быть тяжёлых вычислений: сегментация, черешок
#  и дескриптор запускаются только по нажатию кнопки, результат кладётся
#  в session_state и переиспользуется.
# ═══════════════════════════════════════════════════════════════════════════

tabs = st.tabs([f"{fname} ({len(cuts[fname]['leaves'])})" for fname in names])

for tab, fname in zip(tabs, names):
    with tab:
        leaves = cuts[fname]["leaves"]
        k = len(leaves)

        if k == 0:
            st.warning("Листья не найдены. Проверьте контраст с фоном.")
            continue

        col_scan, col_leaf, col_ctl = st.columns([2, 2, 1.3])

        # ── Слева: скан с подсветкой листьев ────────────────────────────
        with col_scan:
            clicked = leaf_picker(
                scan_bgr=cuts[fname]["scan"],
                contours=leaf_zones(leaves),
                labels=[leaf_name(fname, i) for i in range(k)],
                selected=st.session_state.sel_leaf.get(fname),
                key=f"picker_{fname}",
            )
            st.caption("Кликните по листу, чтобы открыть его справа.")

        if clicked is not None:
            st.session_state.sel_leaf[fname] = clicked

        idx = st.session_state.sel_leaf.get(fname)

        # ── Справа: кнопки и переключатели ──────────────────────────────
        # Каждой операции — кнопка (посчитать) и чекбокс (показать).
        # Чекбокс заблокирован, пока нечего показывать.
        with col_ctl:
            # Сегментация относится ко ВСЕМУ скану, а не к выбранному листу,
            # поэтому кнопка доступна сразу — ещё до того, как лист выбран.
            done_here = sum(1 for i in range(k)
                            if (fname, i) in st.session_state.seg)
            if st.button(f"▶ Сегментировать скан ({done_here}/{k})",
                         key=f"seg_{fname}", width="stretch",
                         type="secondary" if done_here == k else "primary"):
                with st.spinner("Сегментация…"):
                    segment_scan(fname)
                st.rerun()

        # Дальше всё привязано к выбранному листу.
        if idx is None or idx >= k:
            with col_leaf:
                st.info("Лист не выбран — кликните по листу на скане.")
            continue

        leaf = leaves[idx]
        seg_res = st.session_state.seg.get((fname, idx))
        cut_res = st.session_state.petiole.get((fname, idx))
        sym_res = st.session_state.symmetry.get((fname, idx))

        with col_ctl:
            st.markdown(f"**{leaf_name(fname, idx)}**")

            # Пока лист не сегментирован, считать нечего: и черешок, и
            # профиль работают по маске. Держим кнопки заблокированными
            # и объясняем причину в подсказке.
            no_mask = seg_res is None or not seg_res["leaves"]
            need_seg = "Сначала сегментируйте скан" if no_mask else None

            # Бинарный вид — способ смотреть на скан, поэтому положение
            # переключателя общее для всех его листьев (ключ в словаре —
            # имя файла). Если маски ещё нет, чекбокс заблокирован, но
            # значение сохраняется и применится сразу после сегментации.
            bin_key = f"bin_{fname}_{idx}"
            st.checkbox(
                "Бинарный вид",
                value=st.session_state.view_bin.get(fname, False),
                key=bin_key,
                disabled=no_mask, help=need_seg,
                on_change=remember_toggle,
                args=("view_bin", fname, bin_key),
            )
            # Читаем из словаря, а не из возврата чекбокса — см. пояснение
            # в remember_toggle.
            show_binary = st.session_state.view_bin.get(fname, False)

            st.markdown("")

            # 2. Отрезание черешка → разблокирует показ без черешка.
            if st.button("✂ Удалить черенок", key=f"cutbtn_{fname}_{idx}",
                         width="stretch",
                         disabled=no_mask, help=need_seg):
                try:
                    import leaftools
                    with st.spinner("Поиск черенка…"):
                        st.session_state.petiole[(fname, idx)] = \
                            leaftools.cut_petiole(
                                seg_res["leaves"][0]["mask"],
                                svm_model_path=str(MODELS_DIR / "svm_model.xml"),
                                svm_csv_path=str(MODELS_DIR / "SVM_train_7_1.csv"))
                    st.rerun()
                except ImportError:
                    st.error("Модуль leaftools не установлен.")
                except Exception as e:
                    st.error(f"Не удалось отрезать черенок: {e}")

            petiole_ready = cut_res is not None and cut_res.get("found")
            # Отрезание черенка делается для каждого листа отдельно,
            # поэтому и положение переключателя своё у каждого: вернувшись
            # к листу, увидим галочку на месте, а у соседнего, где черенок
            # ещё не отрезан, она снята.
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

            # Маска, на которой считаем признаки: пластина или весь лист.
            leaf_mask = None
            if seg_res is not None and seg_res["leaves"]:
                leaf_mask = seg_res["leaves"][0]["mask"]

            active_mask = leaf_mask
            if hide_petiole and petiole_ready:
                active_mask = cut_res["blade"]

            st.markdown("")

            # 3. Ось симметрии — считается по той маске, что показана.
            if st.button("⟷ Ось симметрии", key=f"symbtn_{fname}_{idx}",
                         width="stretch",
                         disabled=no_mask, help=need_seg):
                try:
                    import leaftools
                    with st.spinner("Поиск оси симметрии…"):
                        sym_res = leaftools.symmetry(active_mask)
                    # Запоминаем, по какой маске считали: подпись под
                    # картинкой покажет, с черенком лист или без.
                    sym_res["without_petiole"] = active_mask is not leaf_mask
                    st.session_state.symmetry[(fname, idx)] = sym_res
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
                help=None if symmetry_ready else "Сначала найдите ось симметрии",
                on_change=remember_toggle,
                args=("view_sym", (fname, idx), sym_key),
            )
            show_symmetry = st.session_state.view_sym.get((fname, idx), False)

            if sym_res is not None and not sym_res.get("found"):
                st.caption("⚠ Ось симметрии не найдена.")

            st.markdown("")

            # 4. Профиль вращения — считается по той маске, что показана.
            want_profile = st.button(
                "📈 Профиль вращения", key=f"prof_{fname}_{idx}",
                width="stretch",
                disabled=no_mask, help=need_seg)

        # ── В центре: выбранный лист в нужном виде ──────────────────────
        # Показ зависит от переключателей:
        #   симметрия     — изображения оси симметрии со стрелками
        #                   переключения; важнее остальных переключателей;
        #   бинарный вид  — маска вместо цветного изображения;
        #   без черенка   — вместо полной маски берётся пластина, а на
        #                   цветном изображении черешок стирается по ней.
        with col_leaf:
            if show_symmetry and symmetry_ready:
                page = st.session_state.sym_page.get((fname, idx), 0)
                view_key, view_title = SYMMETRY_VIEWS[page]
                if sym_res["without_petiole"]:
                    mask_note = "без черенка"
                else:
                    mask_note = "с черенком"
                box_h, box_w = leaf["crop"].shape[:2]
                st.image(rgb(fit_to_box(sym_res[view_key], box_h, box_w)),
                         caption=f"{view_title} · {page + 1}/{len(SYMMETRY_VIEWS)}"
                                 f" · {mask_note}",
                         width="stretch")

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
            elif show_binary and active_mask is not None:
                st.image(active_mask,
                         caption=leaf_name(fname, idx) + " · маска",
                         width="stretch", clamp=True)
            else:
                img = leaf["crop"]
                if hide_petiole and petiole_ready:
                    img = apply_mask_to_color(img, cut_res["blade"])
                st.image(rgb(img), caption=leaf_name(fname, idx),
                         width="stretch")

            if seg_res is None:
                st.caption("Лист не сегментирован — доступен только цветной вид.")

        # ── Под колонками: профиль вращения ─────────────────────────────
        # Кнопка ничего не кэширует: дескриптор считается быстро, а маска
        # зависит от переключателей — проще пересчитать, чем следить.
        if want_profile and active_mask is not None:
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