/**
 * Linux /dev/i2c driver for the Melexis MLX90640 API.
 *
 * Upstream (github.com/melexis/mlx90640-library) ships the calculation API but
 * NO Linux I2C driver, so this supplies the five symbols MLX90640_I2C_Driver.h
 * declares. Compiled as C++ so its symbol names match MLX90640_API.c (which we
 * also build with g++) and Ember's C++ call sites — the upstream headers have
 * no extern "C" guards, so everything must agree on one language.
 *
 * Bus: compile-time default via -DMLX90640_I2C_DEVICE, overridable at runtime
 * with the MLX90640_I2C_BUS env var (Ember calls MLX90640_I2CInit() with no
 * arguments, so bus selection has to happen out-of-band).
 */
#include "MLX90640_I2C_Driver.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/i2c.h>
#include <linux/i2c-dev.h>

#ifndef MLX90640_I2C_DEVICE
#define MLX90640_I2C_DEVICE "/dev/i2c-1"
#endif

static int i2c_fd = -1;

void MLX90640_I2CInit(void)
{
    if (i2c_fd >= 0) {
        return;
    }
    const char* dev = getenv("MLX90640_I2C_BUS");
    if (dev == nullptr || *dev == '\0') {
        dev = MLX90640_I2C_DEVICE;
    }
    // O_CLOEXEC: Ember spawns helper processes; the bus fd must not leak.
    i2c_fd = open(dev, O_RDWR | O_CLOEXEC);
    if (i2c_fd < 0) {
        fprintf(stderr, "[MLX90640] open(%s) failed: %s\n", dev, strerror(errno));
    }
}

int MLX90640_I2CRead(uint8_t slaveAddr, uint16_t startAddress, uint16_t nMemAddressRead, uint16_t* data)
{
    if (i2c_fd < 0) {
        MLX90640_I2CInit();
        if (i2c_fd < 0) {
            return -1;
        }
    }

    uint8_t cmd[2] = {static_cast<uint8_t>(startAddress >> 8), static_cast<uint8_t>(startAddress & 0xFF)};
    // Read straight into the caller's buffer, then byte-swap in place: element i
    // is built from bytes 2i,2i+1 and written back over them, so a forward pass
    // never clobbers unread input. Avoids the fixed 1664-byte bounce buffer.
    uint8_t* raw = reinterpret_cast<uint8_t*>(data);

    struct i2c_msg msgs[2];
    msgs[0].addr = slaveAddr;
    msgs[0].flags = 0;
    msgs[0].len = 2;
    msgs[0].buf = cmd;

    msgs[1].addr = slaveAddr;
    msgs[1].flags = I2C_M_RD;
    msgs[1].len = static_cast<uint16_t>(nMemAddressRead * 2);
    msgs[1].buf = raw;

    struct i2c_rdwr_ioctl_data xfer;
    xfer.msgs = msgs;
    xfer.nmsgs = 2;

    if (ioctl(i2c_fd, I2C_RDWR, &xfer) < 0) {
        return -1; // == -MLX90640_I2C_NACK_ERROR, which the API tests for
    }

    for (uint16_t i = 0; i < nMemAddressRead; ++i) {
        data[i] = static_cast<uint16_t>((static_cast<uint16_t>(raw[2 * i]) << 8) | raw[2 * i + 1]);
    }
    return 0;
}

int MLX90640_I2CWrite(uint8_t slaveAddr, uint16_t writeAddress, uint16_t data)
{
    if (i2c_fd < 0) {
        MLX90640_I2CInit();
        if (i2c_fd < 0) {
            return -1;
        }
    }

    uint8_t cmd[4] = {static_cast<uint8_t>(writeAddress >> 8), static_cast<uint8_t>(writeAddress & 0xFF),
                      static_cast<uint8_t>(data >> 8), static_cast<uint8_t>(data & 0xFF)};

    struct i2c_msg msg;
    msg.addr = slaveAddr;
    msg.flags = 0;
    msg.len = 4;
    msg.buf = cmd;

    struct i2c_rdwr_ioctl_data xfer;
    xfer.msgs = &msg;
    xfer.nmsgs = 1;

    if (ioctl(i2c_fd, I2C_RDWR, &xfer) < 0) {
        return -1;
    }

    // Datasheet: read back and verify the write landed.
    uint16_t check = 0;
    if (MLX90640_I2CRead(slaveAddr, writeAddress, 1, &check) != 0) {
        return -1;
    }
    return (check == data) ? 0 : -2; // == -MLX90640_I2C_WRITE_ERROR
}

int MLX90640_I2CGeneralReset(void)
{
    if (i2c_fd < 0) {
        MLX90640_I2CInit();
        if (i2c_fd < 0) {
            return -1;
        }
    }

    // I2C general call reset: write 0x06 to address 0x00.
    uint8_t cmd = 0x06;
    struct i2c_msg msg;
    msg.addr = 0x00;
    msg.flags = 0;
    msg.len = 1;
    msg.buf = &cmd;

    struct i2c_rdwr_ioctl_data xfer;
    xfer.msgs = &msg;
    xfer.nmsgs = 1;

    return (ioctl(i2c_fd, I2C_RDWR, &xfer) < 0) ? -1 : 0;
}

void MLX90640_I2CFreqSet(int freq)
{
    // No-op. The bus clock is fixed by the kernel driver: dtparam=i2c_arm_baudrate
    // for hardware i2c-1, i2c_gpio_delay_us for the bit-banged i2c-3. There is no
    // per-client ioctl to change it, so Ember's MLX90640_I2CFreqSet(1000) is
    // advisory only. Kept so the API's declared symbol resolves.
    (void)freq;
}
