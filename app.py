import numpy as np
import cv2
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
import altair as alt

from core import segmentation
from core.leaf_splitter import split_leaves

# Конфигурация страницы: заголовок вкладки браузера, иконка, широкий макет.
st.set_page_config(page_title="Анализ листьев", page_icon="🍃", layout="wide")


# ═══════════════════════════════════════════════════════════════════════════
#  Кэшируемые вычисления
#  @st.cache_resource — для тяжёлых объектов (модель), живёт всю сессию.
#  @st.cache_data — по входу: тот же вход не пересчитывается повторно.
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

    leaves = split_leaves(image_bgr, mask_background=True)
    return {"scan": image_bgr, "leaves": leaves}


@st.cache_data(show_spinner=False)
def analyze_leaf(leaf_bytes: bytes):
    """Тяжёлый анализ одного листа: SVM-сегментация."""
    arr = np.frombuffer(leaf_bytes, np.uint8)
    crop = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return segmentation.segment(crop, get_svm())


# ═══════════════════════════════════════════════════════════════════════════
#  Вспомогательные функции обработки изображений
# ═══════════════════════════════════════════════════════════════════════════

def encode(img):
    """Кодирует изображение в PNG-байты (для передачи в кэшируемую функцию)."""
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def rgb(img):
    """OpenCV хранит BGR, Streamlit ждёт RGB — конвертируем перед показом."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def fit_square(img_bgr, size=320, bg=(255, 255, 255)):
    """
    Вписывает изображение в квадрат size×size, сохраняя пропорции,
    с белыми полями. Нужно для ровной сетки превью.
    """
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
    Имя файла для отдельного листа по имени исходного скана.

    Индекс листа (с нуля) превращается в номер (с единицы) и добавляется
    к имени через подчёркивание: "l2nr001.bmp" + 0 -> "l2nr001_1.bmp".
    """
    p = Path(file_name)
    return f"{p.stem}_{leaf_idx + 1}{p.suffix or '.bmp'}"


# ═══════════════════════════════════════════════════════════════════════════
#  БОКОВАЯ ПАНЕЛЬ — глобальные настройки, влияют на все листья сразу.
# ═══════════════════════════════════════════════════════════════════════════

st.sidebar.markdown("### Настройки анализа")

# Отрезать ли черенок перед анализом. Влияет и на профиль вращения,
# и на симметрию: обе считаются по листовой пластине без черенка.
remove_petiole = st.sidebar.checkbox(
    "Удалять черенок",
    value=False,
    help="Отрезать черенок и считать анализ только по листовой пластине.",
)
st.sidebar.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════
#  ШАПКА И ЗАГРУЗКА ФАЙЛОВ
# ═══════════════════════════════════════════════════════════════════════════

st.title("🍃 Анализ листьев")

uploaded = st.file_uploader(
    "Загрузить сканированные изображения листьев",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
    accept_multiple_files=True,
)
st.markdown("---")

# Пока файлов нет — подсказка и остановка скрипта.
if not uploaded:
    st.info("Выберите изображение(я) — листья нарежутся автоматически.")
    st.stop()

# Нарезаем каждый скан на листья (результат кэшируется).
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


# ═══════════════════════════════════════════════════════════════════════════
#  СОСТОЯНИЕ СЕССИИ — сохраняется между перерисовками страницы.
#    analyzed   — результаты сегментации по (файл, индекс листа);
#    selected   — какой лист открыт в крупном виде;
#    sym_cache  — результаты симметрии (тяжёлые!) по (файл, лист, lambda);
#    scroll_*   — служебное для автопрокрутки наверх при выборе листа.
# ═══════════════════════════════════════════════════════════════════════════

if "analyzed" not in st.session_state:
    st.session_state.analyzed = {}
if "selected" not in st.session_state:
    st.session_state.selected = (0, 0)
if "sym_cache" not in st.session_state:
    st.session_state.sym_cache = {}
if "scroll_tick" not in st.session_state:
    st.session_state.scroll_tick = 0
if "scroll_done_at" not in st.session_state:
    st.session_state.scroll_done_at = 0

# Чистим результаты для файлов, которых больше нет среди загруженных.
valid = set(cuts.keys())
st.session_state.analyzed = {
    k: v for k, v in st.session_state.analyzed.items() if k[0] in valid}


def do_analyze(fname, idx, crop):
    """Анализ одного листа; результат кладём в session_state."""
    st.session_state.analyzed[(fname, idx)] = analyze_leaf(encode(crop))


# ═══════════════════════════════════════════════════════════════════════════
#  ПАНЕЛЬ ПРОГРЕССА — кнопка «проанализировать всё» и индикатор готовности.
# ═══════════════════════════════════════════════════════════════════════════

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
# Невидимый якорь — к нему прокручиваем страницу при выборе листа.
st.markdown('<div id="top-anchor"></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  ВЫБОР ТЕКУЩЕГО ЛИСТА (страхуемся от выхода индексов за границы)
# ═══════════════════════════════════════════════════════════════════════════

names = list(cuts.keys())
sel_f, sel_i = st.session_state.selected
if sel_f >= len(names):
    sel_f, sel_i = 0, 0
sel_fname = names[sel_f]
if sel_i >= len(cuts[sel_fname]["leaves"]):
    sel_i = 0
st.session_state.selected = (sel_f, sel_i)


# ═══════════════════════════════════════════════════════════════════════════
#  КРУПНЫЙ ВИД ВЫБРАННОГО ЛИСТА
#    слева  — большое изображение листа;
#    справа — признаки, отрезание черенка и вкладки анализа.
# ═══════════════════════════════════════════════════════════════════════════

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

        # Результат тяжёлого анализа — только после нажатия кнопки.
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

            # ── Отрезание черенка (управляется галочкой в сайдбаре) ──────
            # Если галочка включена — режем маску на пластину и черенок.
            # Дальше ВЕСЬ анализ (профиль и симметрия) идёт по пластине.
            leaf_mask = seg["mask"]        # лист = 255 на чёрном
            analysis_mask = leaf_mask      # по умолчанию — вся маска

            if remove_petiole:
                try:
                    import leaftools
                    models_dir = Path(__file__).parent / "models"
                    svm_model = str(models_dir / "svm_model.xml")
                    svm_csv = str(models_dir / "SVM_train_7_1.csv")

                    cut = leaftools.cut_petiole(
                        leaf_mask, svm_model_path=svm_model,
                        svm_csv_path=svm_csv)

                    if cut["found"]:
                        analysis_mask = cut["blade"]
                        cc1, cc2 = st.columns(2)
                        with cc1:
                            st.image(cut["blade"], caption="Пластина",
                                     use_container_width=True, clamp=True)
                        with cc2:
                            st.image(cut["petiole"], caption="Черенок",
                                     use_container_width=True, clamp=True)
                        st.caption("Анализ — по листовой пластине.")
                    else:
                        st.warning("Точки черенка не найдены — "
                                   "анализ по всей маске.")
                except ImportError:
                    st.info("Модуль leaftools не установлен.")
                except Exception as e:
                    st.warning(f"Не удалось отрезать черенок: {e}")

            # ── ВКЛАДКИ АНАЛИЗА ─────────────────────────────────────────
            # Каждый вид анализа на своей вкладке. Считается только то,
            # что открыто, поэтому тяжёлые вещи не мешают лёгким.
            tab_profile, tab_symmetry, tab_classify = st.tabs(
                ["Профиль вращения", "Симметрия", "Классификация"]
            )

            # ── Вкладка 1: профиль вращения (быстрый, считаем сразу) ────
            with tab_profile:
                try:
                    import leaftools
                    desc = leaftools.describe_mask(analysis_mask)

                    # График: угол (0…180) -> мера Жаккара.
                    # Оси зафиксированы, чтобы вид не искажался.
                    chart_df = pd.DataFrame({
                        "angle": desc["angles"],
                        "jaccard": desc["jaccard_values"],
                    })
                    chart = (
                        alt.Chart(chart_df)
                        .mark_line()
                        .encode(
                            x=alt.X("angle:Q",
                                    scale=alt.Scale(domain=[0, 180]),
                                    title="Угол"),
                            y=alt.Y("jaccard:Q",
                                    scale=alt.Scale(domain=[0, 1]),
                                    title="Жаккар"),
                        )
                        .properties(height=200)
                    )
                    st.altair_chart(chart, use_container_width=True)

                    # Выгрузка дескриптора в CSV (разделитель ';').
                    csv = "angle;jaccard\n" + "\n".join(
                        f"{int(a)};{v:.6f}"
                        for a, v in zip(desc["angles"], desc["jaccard_values"])
                    )
                    st.download_button(
                        "Скачать дескриптор (CSV)",
                        data=csv,
                        file_name=f"{sel_label}_descriptor.csv",
                        mime="text/csv",
                        key=f"desc_csv_{sel_f}_{sel_i}",
                    )
                except ImportError:
                    st.info("Модуль leaftools не установлен. "
                            "Соберите его из папки leaftools_cpp/.")
                except Exception as e:
                    st.warning(f"Не удалось посчитать дескриптор: {e}")

            # ── Вкладка 2: симметрия (ТЯЖЁЛАЯ — только по кнопке) ───────
            with tab_symmetry:
                # λ задаётся здесь же: это параметр именно симметрии.
                lam = st.slider(
                    "λ — сглаживание оси", 0.0, 1.0, 0.5, 0.05,
                    key=f"lam_{sel_f}_{sel_i}",
                    help="Больше λ — ось глаже и ближе к предыдущему "
                         "направлению; меньше — ось точнее следует форме.",
                )

                # Ключ кэша включает λ: сменили λ — нужен новый расчёт.
                sym_key = (sel_fname, sel_i, round(lam, 2), remove_petiole)

                if st.button("Найти симметрию", key=f"sym_btn_{sel_f}_{sel_i}"):
                    try:
                        import leaftools
                        with st.spinner("Поиск оси симметрии… это долго"):
                            st.session_state.sym_cache[sym_key] = \
                                leaftools.symmetry(analysis_mask, lambda_=lam)
                    except ImportError:
                        st.info("Модуль leaftools не установлен.")
                    except Exception as e:
                        st.warning(f"Не удалось найти симметрию: {e}")

                sym = st.session_state.sym_cache.get(sym_key)

                if sym is None:
                    st.info("Нажмите «Найти симметрию». "
                            "Расчёт занимает заметное время.")
                elif not sym.get("found"):
                    st.warning("Ось симметрии не найдена.")
                else:
                    m1, m2 = st.columns(2)
                    m1.metric("Жаккар (распрямл.)",
                              f"{sym['jaccard_straightened']:.3f}")
                    m2.metric("Жаккар (исходн.)",
                              f"{sym['jaccard_original']:.3f}")

                    # Картинки приходят из C++ в BGR — указываем это явно.
                    st.image(sym["axis"], caption="Ось симметрии",
                             use_container_width=True, channels="BGR")
                    s1, s2 = st.columns(2)
                    with s1:
                        st.image(sym["straightened"], caption="Распрямлённый",
                                 use_container_width=True, channels="BGR")
                    with s2:
                        st.image(sym["reflected"], caption="Отражённый",
                                 use_container_width=True, channels="BGR")

            # ── Вкладка 3: классификация (заготовка) ────────────────────
            with tab_classify:
                st.info("Классификация — в разработке.")
else:
    with big_col:
        st.warning("На этом скане листья не найдены.")

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════
#  ГАЛЕРЕЯ ПО СКАНАМ — на каждый файл: слева исходный скан,
#  справа сетка превью листьев с кнопками выбора.
# ═══════════════════════════════════════════════════════════════════════════

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
                GRID = 3  # листьев в ряду сетки
                for row in range(0, k, GRID):
                    cols = st.columns(GRID)
                    for cidx in range(GRID):
                        i = row + cidx
                        if i >= k:
                            break
                        with cols[cidx]:
                            st.image(rgb(fit_square(c["leaves"][i]["crop"], 320)),
                                     use_container_width=True)
                            # Кнопка выбирает лист в крупный вид.
                            # ✓ — лист уже проанализирован.
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


# ═══════════════════════════════════════════════════════════════════════════
#  СВОДНАЯ ТАБЛИЦА ПРИЗНАКОВ по всем листьям + выгрузка в CSV.
#  _bbox — признаки быстрой нарезки; _seg — признаки сегментации.
# ═══════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════
#  АВТОПРОКРУТКА НАВЕРХ при выборе нового листа.
#  tick в скрипте уникализирует его — иначе браузер выполнит вставку раз.
# ═══════════════════════════════════════════════════════════════════════════

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