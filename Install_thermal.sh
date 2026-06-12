#!/bin/sh
# =============================================================================
# Ember — MLX90640-D55 thermal camera setup
#
# Enables I2C, raises the bus to 1 MHz (needed for stable >=8 Hz reads), and
# builds the Melexis MLX90640 driver as a shared library + headers so the CMake
# build can auto-detect it (EMBER_HAVE_MLX90640) and enable the thermal overlay,
# fire warning, and --detector-source thermal.
#
# Run AFTER Install_dependencies.sh, then rebuild with ./compile.sh.
# Wiring: MLX90640 VIN->3V3, GND->GND, SDA->GPIO2 (pin 3), SCL->GPIO3 (pin 5).
# Default I2C address: 0x33.
# =============================================================================
set -e

echo "== Enabling I2C"
if command -v raspi-config >/dev/null 2>&1; then
    sudo raspi-config nonint do_i2c 0 || true
fi

# Raise the ARM I2C baudrate to 1 MHz for reliable thermal frame reads.
CONFIG_TXT="/boot/firmware/config.txt"
[ -f "$CONFIG_TXT" ] || CONFIG_TXT="/boot/config.txt"
if [ -f "$CONFIG_TXT" ]; then
    if ! grep -q "dtparam=i2c_arm_baudrate=1000000" "$CONFIG_TXT"; then
        echo "== Setting I2C baudrate to 1 MHz in $CONFIG_TXT"
        sudo sed -i '/dtparam=i2c_arm_baudrate=/d' "$CONFIG_TXT"
        echo "dtparam=i2c_arm=on" | sudo tee -a "$CONFIG_TXT" >/dev/null
        echo "dtparam=i2c_arm_baudrate=1000000" | sudo tee -a "$CONFIG_TXT" >/dev/null
    fi
fi

echo "== Installing I2C + build tooling"
sudo apt-get update
sudo apt-get install -y i2c-tools build-essential git

echo "== Building Melexis MLX90640 driver (libMLX90640.so)"
WORK="/tmp/mlx90640-library"
rm -rf "$WORK"
git clone --depth 1 https://github.com/melexis/mlx90640-library "$WORK"
# Compile the calculation API + the Linux /dev/i2c driver into one shared lib.
g++ -std=c++11 -O2 -fPIC -shared -I"$WORK/headers" \
    "$WORK/functions/MLX90640_API.cpp" \
    "$WORK/functions/MLX90640_LINUX_I2C_Driver.cpp" \
    -o "$WORK/libMLX90640.so"

sudo install -d /usr/local/lib /usr/local/include/MLX90640
sudo install -m 0644 "$WORK/libMLX90640.so" /usr/local/lib/
sudo install -m 0644 "$WORK"/headers/*.h /usr/local/include/MLX90640/
sudo ldconfig

echo ""
echo "== Thermal setup complete."
echo "   Verify the sensor:  i2cdetect -y 1   (expect a device at 0x33)"
echo "   Rebuild:            ./compile.sh     (look for 'MLX90640 thermal support: ENABLED')"
echo "   The thermal overlay + fire warning are on by default; for thermal-primary"
echo "   victim detection add:  --detector-source thermal"
echo "   NOTE: the 1 MHz I2C baudrate change requires a reboot to take effect."
