#!/bin/sh
# =============================================================================
# Ember — MLX90640-D55 thermal camera setup
#
# Builds the Melexis MLX90640 calculation API plus Ember's Linux /dev/i2c driver
# into libMLX90640.so so the CMake build auto-detects it (EMBER_HAVE_MLX90640)
# and enables the thermal overlay, fire warning, and --detector-source thermal.
#
# Upstream (github.com/melexis/mlx90640-library) ships ONLY functions/MLX90640_API.c
# and headers/ — there is no Linux I2C driver in it at all. Ember supplies one at
# src/cpp/vendor/MLX90640_LINUX_I2C_Driver.cpp.
#
# Run AFTER Install_dependencies.sh, then rebuild with ./compile.sh.
# Wiring on this unit: SDA->GPIO27, SCL->GPIO22, bit-banged bus 3 (see config.txt
# dtoverlay=i2c-gpio,bus=3,...). Default I2C address 0x33.
# =============================================================================
set -e

REPO_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

# Which /dev/i2c-N the sensor is on. Bus 3 is the bit-banged i2c-gpio bus that
# the MLX90640 is actually wired to on this unit; nothing is on hardware bus 1.
# Overridable at runtime with the MLX90640_I2C_BUS env var.
I2C_BUS_DEV="${MLX90640_I2C_BUS:-/dev/i2c-3}"

echo "== Installing I2C + build tooling"
sudo apt-get update
sudo apt-get install -y i2c-tools build-essential git

echo "== Building MLX90640 driver (libMLX90640.so) for $I2C_BUS_DEV"
WORK="/tmp/mlx90640-library"
rm -rf "$WORK"
git clone --depth 1 https://github.com/melexis/mlx90640-library "$WORK"

# Both halves are compiled with g++. The upstream headers have NO extern "C"
# guards, so the library and Ember's C++ call sites must agree on one language;
# building the API as C++ makes the mangled names match with no header patching.
g++ -std=c++11 -O2 -fPIC -x c++ -I"$WORK/headers" \
    -c "$WORK/functions/MLX90640_API.c" -o "$WORK/MLX90640_API.o"

g++ -std=c++11 -O2 -fPIC -x c++ -I"$WORK/headers" \
    -DMLX90640_I2C_DEVICE="\"$I2C_BUS_DEV\"" \
    -c "$REPO_DIR/src/cpp/vendor/MLX90640_LINUX_I2C_Driver.cpp" \
    -o "$WORK/MLX90640_LINUX_I2C_Driver.o"

# --no-undefined turns a missing driver symbol into a build error here rather
# than a confusing link failure when Ember itself is built.
g++ -shared -Wl,--no-undefined \
    "$WORK/MLX90640_API.o" "$WORK/MLX90640_LINUX_I2C_Driver.o" \
    -o "$WORK/libMLX90640.so"

sudo install -d /usr/local/lib /usr/local/include/MLX90640
sudo install -m 0644 "$WORK/libMLX90640.so" /usr/local/lib/
sudo install -m 0644 "$WORK"/headers/*.h /usr/local/include/MLX90640/
sudo ldconfig

echo ""
echo "== Thermal setup complete."
echo "   Verify the sensor:  i2cdetect -y 3   (expect a device at 0x33)"
echo "   Rebuild:            ./compile.sh     (look for 'MLX90640 thermal support: ENABLED')"
echo "   Override the bus at runtime: MLX90640_I2C_BUS=/dev/i2c-N ./run.sh"
echo ""
echo "   NOTE: bus 3 is bit-banged (i2c-gpio). A thermal frame costs ~63 ms of"
echo "   busy-wait CPU, so 8 Hz burns most of a core. This Pi already throttles"
echo "   at 83-85 C — start with --thermal-refresh 2 and raise it only if the"
echo "   HUD frame rate and SoC temperature hold up."
