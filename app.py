import numpy as np
import cv2
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

from core import segmentation
from core.leaf_splitter import split_leaves

st.set_page_config(page_title="Анализ листьев", page_icon="🍃", layout="wide")

@st.cache_resource(show_spinner="Обучение модели фон/лист…")
def get_svm():
    return segmentation.load_svm()


@st.cache_data(show_spinner=False)
def cut_leaves(image_bytes: bytes):
    arr = np.frombuffer(image_bytes, np.uint8)
    image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return None

    leaves = split_leaves(image_bgr, mask_background=True)
    return {"scan": image_bgr, "leaves": leaves}


@st.cache_data(show_spinner=False)
def analyze_leaf(leaf_bytes: bytes):
    arr = np.frombuffer(leaf_bytes, np.uint8)
    crop = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return segmentation.segment(crop, get_svm())


def encode(img):
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def fit_square(img_bgr, size=320, bg=(255, 255, 255)):
    h, w = img_bgr.shape[:2]
    scale = size / max(h, w)
    nw, nh = max(int(round(w * scale)), 1), max(int(round(h * scale)), 1)
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.full((size, size, 3), bg, np.uint8)
    y0 = (size - nh) // 2
    x0 = (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas

def leaf_name(file_name: str, leaf_idx: int) -> str:
    """
    Формирует имя файла для отдельного листа по имени исходного скана.

    Индекс листа (с нуля) превращается в номер (с единицы) и добавляется
    к имени через подчёркивание. Расширение сохраняется; если его нет,
    подставляется .bmp.

    Параметры
    ---------
    file_name : str
        Имя исходного файла скана, например "l2nr001.bmp".
    leaf_idx : int
        Индекс листа, начиная с 0.

    Возвращает
    ----------
    str
        Имя листа, например "l2nr001_1.bmp" для leaf_idx=0.
    """
    p = Path(file_name)
    return f"{p.stem}_{leaf_idx + 1}{p.suffix or '.bmp'}"

st.sidebar.markdown("### Настройки анализа")
st.sidebar.markdown("---")

st.title("🍃 Анализ листьев")
uploaded = st.file_uploader(
    "Загрузить сканированные изображения листьев",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
    accept_multiple_files=True,
)
st.markdown("---")
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

total_leaves = sum(len(c["leaves"]) for c in cuts.values())
st.success(f"Загружено сканов: {len(cuts)} · нарезано листьев: {total_leaves}")

if "analyzed" not in st.session_state:
    st.session_state.analyzed = {}
if "selected" not in st.session_state:
    st.session_state.selected = (0, 0)
if "scroll_tick" not in st.session_state:
    st.session_state.scroll_tick = 0
if "scroll_done_at" not in st.session_state:
    st.session_state.scroll_done_at = 0

valid = set(cuts.keys())
st.session_state.analyzed = {
    k: v for k, v in st.session_state.analyzed.items() if k[0] in valid}


def do_analyze(fname, idx, crop):
    st.session_state.analyzed[(fname, idx)] = analyze_leaf(encode(crop))

n_done = len(st.session_state.analyzed)
top_l, top_r = st.columns([1, 3])
with top_l:
    if st.button("▶ Проанализировать всё", type="primary",
                 use_container_width=True):
        bar = st.progress(0.0, text="Анализ…")
        done = 0
        for fname, c in cuts.items():
            for i, leaf in enumerate(c["leaves"]):
                bar.progress(done / max(total_leaves, 1),
                             text=f"Анализ: {leaf_name(fname, i)}")
                do_analyze(fname, i, leaf["crop"])
                done += 1
        bar.empty()
        st.rerun()
with top_r:
    st.progress(n_done / max(total_leaves, 1),
                text=f"Проанализировано листьев: {n_done} из {total_leaves}")

st.markdown("---")
st.markdown('<div id="top-anchor"></div>', unsafe_allow_html=True)

names = list(cuts.keys())
sel_f, sel_i = st.session_state.selected
if sel_f >= len(names):
    sel_f, sel_i = 0, 0
sel_fname = names[sel_f]
if sel_i >= len(cuts[sel_fname]["leaves"]):
    sel_i = 0
st.session_state.selected = (sel_f, sel_i)

big_col, info_col = st.columns([2, 1])

if cuts[sel_fname]["leaves"]:
    sel_leaf = cuts[sel_fname]["leaves"][sel_i]
    sel_label = leaf_name(sel_fname, sel_i)
    sel_result = st.session_state.analyzed.get((sel_fname, sel_i))

    with big_col:
        st.image(rgb(fit_square(sel_leaf["crop"], 480)), caption=sel_label,
                 use_container_width=True)

    with info_col:
        st.subheader("Лист")
        st.write(f"**Файл:** `{sel_label}`")

        # Признаки быстрой нарезки — доступны сразу
        st.caption("Признаки контура (быстрая нарезка)")
        st.dataframe(pd.DataFrame([{
            "area": round(sel_leaf["area"], 1),
            "perimeter": round(sel_leaf["perimeter"], 1),
            "form_factor": round(sel_leaf["form_factor"], 3),
        }]), use_container_width=True, hide_index=True)

        # Результат тяжёлого анализа — только после кнопки
        if sel_result is None:
            st.info("Лист не проанализирован. Нажмите «Проанализировать».")
        elif not sel_result["leaves"]:
            st.warning("Сегментация не нашла лист на вырезе.")
        else:
            seg = sel_result["leaves"][0]
            st.caption("Признаки сегментации (SVM)")
            st.dataframe(pd.DataFrame([{
                "area": round(seg["area"], 1),
                "perimeter": round(seg["perimeter"], 1),
                "form_factor": round(seg["form_factor"], 3),
            }]), use_container_width=True, hide_index=True)
            st.image(sel_result["binary"], caption="Маска сегментации",
                     use_container_width=True, clamp=True)
else:
    with big_col:
        st.warning("На этом скане листья не найдены.")

st.markdown("---")

for file_idx, fname in enumerate(names):
    c = cuts[fname]
    k = len(c["leaves"])
    done_here = sum(1 for i in range(k) if (fname, i) in st.session_state.analyzed)

    head_l, head_r = st.columns([3, 1])
    with head_l:
        st.markdown(f"#### {fname}  ·  листьев: {k}  ·  "
                    f"проанализировано: {done_here}/{k}")
    with head_r:
        if k > 0 and st.button("▶ Проанализировать скан", key=f"an_{file_idx}",
                               use_container_width=True,
                               type="secondary" if done_here == k else "primary"):
            with st.spinner(f"Анализ {fname}…"):
                for i, leaf in enumerate(c["leaves"]):
                    do_analyze(fname, i, leaf["crop"])
            st.rerun()

    left, right = st.columns([1, 2])

    with left:
        st.image(rgb(c["scan"]), caption="Исходный скан",
                 use_container_width=True)

    with right:
        with st.container(border=True):
            if k == 0:
                st.warning("Листья не найдены. Проверьте контраст с фоном.")
            else:
                GRID = 3
                for row in range(0, k, GRID):
                    cols = st.columns(GRID)
                    for cidx in range(GRID):
                        i = row + cidx
                        if i >= k:
                            break
                        with cols[cidx]:
                            st.image(rgb(fit_square(c["leaves"][i]["crop"], 320)),
                                     use_container_width=True)
                            is_sel = (file_idx, i) == tuple(st.session_state.selected)
                            mark = "✓ " if (fname, i) in st.session_state.analyzed else ""
                            if st.button(f"{mark}{leaf_name(fname, i)}",
                                         key=f"b_{file_idx}_{i}",
                                         use_container_width=True,
                                         type="primary" if is_sel else "secondary"):
                                st.session_state.selected = (file_idx, i)
                                st.session_state.scroll_tick += 1
                                st.rerun()

    st.markdown("---")

rows = []
for fname in names:
    for i, leaf in enumerate(cuts[fname]["leaves"]):
        res = st.session_state.analyzed.get((fname, i))
        seg = res["leaves"][0] if (res and res["leaves"]) else None
        rows.append({
            "файл": leaf_name(fname, i),
            "area_bbox": round(leaf["area"], 1),
            "perimeter_bbox": round(leaf["perimeter"], 1),
            "form_factor_bbox": round(leaf["form_factor"], 3),
            "area_seg": round(seg["area"], 1) if seg else None,
            "perimeter_seg": round(seg["perimeter"], 1) if seg else None,
            "form_factor_seg": round(seg["form_factor"], 3) if seg else None,
        })

if rows:
    df = pd.DataFrame(rows)
    st.subheader("Все признаки")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button(
        "Скачать признаки (CSV)",
        data=df.to_csv(index=False, sep=";").encode("utf-8"),
        file_name="leaf_features.csv", mime="text/csv")

if st.session_state.scroll_tick > st.session_state.scroll_done_at:
    st.session_state.scroll_done_at = st.session_state.scroll_tick
    tick = st.session_state.scroll_tick
    components.html(
        f"""
        <script>
            // tick={tick} — уникализирует скрипт, иначе сработает лишь раз
            const doc = window.parent.document;
            const a = doc.getElementById("top-anchor");
            if (a) {{ a.scrollIntoView({{behavior: "smooth", block: "start"}}); }}
            else {{ window.parent.scrollTo({{top: 0, behavior: "smooth"}}); }}
        </script>
        """, height=0)