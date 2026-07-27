#!/usr/bin/env bash
# ===========================================================================
# Build leaftools on Linux.
#
# Run from the leaf_cpp folder:  ./build.sh
# Requires: active venv, vcpkg (VCPKG_ROOT set or edit the path below).
#
# On Linux there is NO delvewheel step — the .so finds its shared libraries
# through the standard mechanism, and auditwheel (used only for distributable
# wheels) is not needed for local use.
# ===========================================================================

set -e  # stop on first error

# Path to vcpkg. Uses $VCPKG_ROOT if set, otherwise this default.
VCPKG_ROOT="${VCPKG_ROOT:-$HOME/vcpkg}"
echo "Using VCPKG_ROOT=$VCPKG_ROOT"

VCPKG_TOOLCHAIN="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"

# Clean previous build output (keep vcpkg cache).
rm -rf wheelhouse

echo
echo "=== Step 1: build wheel ==="
pip wheel . -w wheelhouse --no-deps \
  --config-settings=cmake.define.CMAKE_TOOLCHAIN_FILE="$VCPKG_TOOLCHAIN" \
  --config-settings=cmake.define.VCPKG_TARGET_TRIPLET=x64-linux \
  --config-settings=cmake.define.VCPKG_MANIFEST_MODE=ON

echo
echo "=== Step 2: install wheel ==="
pip install --force-reinstall wheelhouse/*.whl

echo
echo "=== Done ==="
echo "Check: python -c \"import leaftools; print(leaftools.cv_version())\""
