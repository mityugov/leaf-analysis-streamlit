# Анализ листьев

Streamlit-приложение для анализа сканов листьев: нарезка скана на отдельные
листья, сегментация лист/фон, отрезание черешка, профиль вращения и ось
симметрии.

Часть алгоритмов написана на C++ и собирается в Python-модуль `leaftools`
(папка `leaftools_cpp/`). Приложение работает и без него, но тогда недоступны
черешок, профиль вращения и симметрия.

## Структура

```
app.py              — Streamlit-приложение
core/               — нарезка скана и сегментация (Python)
components/         — компонент выбора листа на скане
models/             — модель SVM для поиска черешка
assets/             — данные для сегментации
leaftools_cpp/      — C++ модуль leaftools (pybind11 + OpenCV)
```

## Что нужно

- **Python 3.11.** Скрипт `build.bat` рассчитан на эту версию (см. ниже,
  как собрать под другую).
- **C++ компилятор:**
  - Windows — Visual Studio 2022 или Build Tools for Visual Studio 2022
    с компонентом «Разработка классических приложений на C++»;
  - Linux — GCC (пакет `build-essential`).
- **OpenCV для C++.** Ставится через [vcpkg](https://github.com/microsoft/vcpkg);
  на Linux можно взять системный пакет.

CMake отдельно ставить не обязательно: при сборке `pip` скачает его сам.

## Сборка на Windows

Команды выполняются в «Developer PowerShell for VS 2022» или в обычном
терминале из корня репозитория.

### 1. Виртуальное окружение и зависимости

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. vcpkg (один раз)

```powershell
git clone https://github.com/microsoft/vcpkg E:\vcpkg
E:\vcpkg\bootstrap-vcpkg.bat
```

Папку можно выбрать любую. Если она не `E:\vcpkg`, поправьте путь
в `leaftools_cpp\build.bat`:

```bat
set VCPKG_TOOLCHAIN=E:/vcpkg/scripts/buildsystems/vcpkg.cmake
```

Путь пишется с прямыми слешами `/`.

### 3. Сборка модуля leaftools

```powershell
cd leaftools_cpp
.\build.bat
cd ..
```

Скрипт делает три шага:

1. собирает wheel (`pip wheel`); vcpkg при этом скачивает и собирает OpenCV
   и pybind11 по `vcpkg.json`;
2. упаковывает DLL OpenCV внутрь wheel через `delvewheel`, иначе модуль
   не найдёт их при импорте;
3. устанавливает wheel в активное окружение.

Первая сборка долгая (OpenCV компилируется из исходников, до получаса и
больше). Следующие сборки быстрые: собранные библиотеки остаются
в `leaftools_cpp\build\`.

**Другая версия Python.** В `build.bat` указан тег `cp311`, по нему скрипт
ищет DLL. Для Python 3.12 замените его:

```bat
set WHEEL_TAG=cp312-cp312-win_amd64
```

### 4. Проверка

```powershell
python -c "import leaftools; print(leaftools.cv_version())"
```

Должна вывестись версия OpenCV, с которой собран модуль.

## Сборка на Linux

### 1. Системные пакеты

Ubuntu / Debian:

```bash
sudo apt install build-essential git curl zip unzip tar pkg-config python3.11 python3.11-venv
```

Если в системе нет Python 3.11 (например, Ubuntu 22.04), его можно поставить
из PPA `deadsnakes`. Версии из `requirements.txt` требуют Python 3.11 или новее.

### 2. Виртуальное окружение и зависимости

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Сборка модуля leaftools

Есть два способа.

**Способ А — через vcpkg**, как на Windows и в CI. OpenCV собирается
статически и попадает внутрь модуля.

```bash
git clone https://github.com/microsoft/vcpkg ~/vcpkg
~/vcpkg/bootstrap-vcpkg.sh

cd leaftools_cpp
./build.sh
cd ..
```

Скрипт берёт vcpkg из `$VCPKG_ROOT`, а если переменная не задана —
из `~/vcpkg`. Первая сборка долгая, как и на Windows.

**Способ Б — системный OpenCV.** Быстрее, но модуль будет зависеть от
библиотек OpenCV, установленных в системе, и работать только там, где они есть.

```bash
sudo apt install libopencv-dev

cd leaftools_cpp
pip wheel . -w wheelhouse --no-deps
pip install --force-reinstall wheelhouse/*.whl
cd ..
```

### 4. Проверка

```bash
python -c "import leaftools; print(leaftools.cv_version())"
```

## Тесты модуля

```bash
pip install pytest
python -m pytest leaftools_cpp/tests -q
```

## Запуск приложения

Из корня репозитория, с активированным окружением:

```bash
streamlit run app.py
```

Адрес и порт заданы в `.streamlit/config.toml`. Приложение открывается по
адресу <http://127.0.0.1:8503/leaves>.

## Если что-то не так

| Сообщение | Причина |
|---|---|
| «Модуль leaftools не установлен» в приложении | модуль не собран или установлен в другое окружение: проверьте, что `.venv` активирован и при сборке, и при запуске |
| `ImportError: DLL load failed` (Windows) | wheel установлен без шага `delvewheel`: соберите через `build.bat`, а не командой `pip wheel` напрямую |
| `Could not find a package configuration file provided by "OpenCV"` | CMake не нашёл OpenCV: не указан путь к vcpkg (`VCPKG_TOOLCHAIN` в `build.bat`, `VCPKG_ROOT` для `build.sh`) или на Linux не установлен `libopencv-dev` |
| `WARNING: DLLs not found` при сборке (Windows) | `WHEEL_TAG` в `build.bat` не совпадает с версией Python |
