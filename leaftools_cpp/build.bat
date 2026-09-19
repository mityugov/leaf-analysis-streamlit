@echo off
setlocal enabledelayedexpansion
REM ===========================================================================
REM Build leaftools - three steps in one command.
REM
REM Run from the leaf_cpp folder:  build.bat
REM Requires: active venv, vcpkg at the path below.
REM ===========================================================================

REM Path to vcpkg toolchain - use FORWARD slashes (%VAR:\=/% is unreliable).
set VCPKG_TOOLCHAIN=E:/vcpkg/scripts/buildsystems/vcpkg.cmake

REM Wheel tag (Python 3.11, win64) - used to locate the DLL folder.
set WHEEL_TAG=cp311-cp311-win_amd64

echo === leaftools build ===

REM Clean previous wheel and fixed output. Keep build\ and vcpkg_installed\ (cache).
if exist wheelhouse rmdir /s /q wheelhouse
if exist fixed rmdir /s /q fixed

echo.
echo === Step 1: build wheel ===
pip wheel . -w wheelhouse --no-deps ^
  --config-settings=cmake.define.CMAKE_TOOLCHAIN_FILE=%VCPKG_TOOLCHAIN% ^
  --config-settings=cmake.define.VCPKG_TARGET_TRIPLET=x64-windows ^
  --config-settings=cmake.define.VCPKG_MANIFEST_MODE=ON
if errorlevel 1 goto :error

echo.
echo === Step 2: delvewheel repair (bundle OpenCV DLLs) ===
pip install delvewheel >nul 2>&1

REM DLLs live in build\<tag>\vcpkg_installed\x64-windows\bin (manifest mode).
set DLL_DIR=build\%WHEEL_TAG%\vcpkg_installed\x64-windows\bin
if not exist "%DLL_DIR%\opencv_core4.dll" (
    echo WARNING: DLLs not found in %DLL_DIR%
    echo Searching for opencv_core4.dll elsewhere...
    for /f "delims=" %%p in ('dir /s /b build\*\vcpkg_installed\*\opencv_core4.dll 2^>nul') do set DLL_DIR=%%~dpp
)
echo DLL dir: %DLL_DIR%

for %%w in (wheelhouse\*.whl) do (
    delvewheel repair "%%w" -w fixed --add-path "%DLL_DIR%"
    if errorlevel 1 goto :error
)

echo.
echo === Step 3: install repaired wheel ===
for %%w in (fixed\*.whl) do (
    pip install --force-reinstall "%%w"
    if errorlevel 1 goto :error
)

echo.
echo === Done ===
echo Check: python -c "import leaftools; print(leaftools.cv_version())"
goto :eof

:error
echo.
echo !!! Build failed. See output above.
exit /b 1