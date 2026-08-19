#!/usr/bin/env python3

import argparse
import sys
import time

from typing import Any

try:
    import lgpio
except ImportError:  # dev box without the hardware lib: enough for --selfcheck
    lgpio: Any = type("lgpio", (), {"error": RuntimeError})

DHTXX = 2

DHT_GOOD = 0
DHT_BAD_CHECKSUM = 1
DHT_BAD_DATA = 2
DHT_TIMEOUT = 3

STATUS_LABELS = {
    DHT_GOOD: "GOOD",
    DHT_BAD_CHECKSUM: "BAD_CHECKSUM",
    DHT_BAD_DATA: "BAD_DATA",
    DHT_TIMEOUT: "TIMEOUT",
}


class DHTSensor:
    def __init__(self, chip, gpio, model=DHTXX):
        self._chip = chip
        self._gpio = gpio
        self._model = model
        self._new_data = False
        self._bits = 0
        self._code = 0
        self._last_edge_tick = 0
        self._status = DHT_TIMEOUT
        self._temperature = 0.0
        self._humidity = 0.0

        lgpio.gpio_set_watchdog_micros(chip, gpio, 1000)
        self._cb = lgpio.callback(chip, gpio, lgpio.RISING_EDGE, self._rising_edge)

    def cancel(self):
        if self._cb is not None:
            self._cb.cancel()
            self._cb = None

    def _validate_dhtxx(self, b1, b2, b3, b4):
        div = -10.0 if (b2 & 128) else 10.0
        temperature_c = float(((b2 & 127) << 8) + b1) / div
        humidity_percent = float((b4 << 8) + b3) / 10.0
        valid = (humidity_percent <= 110.0) and (-50.0 <= temperature_c <= 135.0)
        return valid, temperature_c, humidity_percent

    def _decode(self):
        b0 = self._code & 0xFF
        b1 = (self._code >> 8) & 0xFF
        b2 = (self._code >> 16) & 0xFF
        b3 = (self._code >> 24) & 0xFF
        b4 = (self._code >> 32) & 0xFF
        checksum = (b1 + b2 + b3 + b4) & 0xFF

        if checksum != b0:
            self._status = DHT_BAD_CHECKSUM
            self._new_data = True
            return

        valid, temperature_c, humidity_percent = self._validate_dhtxx(b1, b2, b3, b4)
        if valid:
            self._temperature = temperature_c
            self._humidity = humidity_percent
            self._status = DHT_GOOD
        else:
            self._status = DHT_BAD_DATA
        self._new_data = True

    def _rising_edge(self, chip, gpio, level, tick):
        if level != lgpio.TIMEOUT:
            edge_len = tick - self._last_edge_tick
            self._last_edge_tick = tick
            if edge_len > 2e8:
                self._bits = 0
                self._code = 0
            else:
                self._code <<= 1
                if edge_len > 1e5:
                    self._code |= 1
                self._bits += 1
        else:
            if self._bits >= 30:
                self._decode()

    def _free(self):
        try:
            lgpio.gpio_free(self._chip, self._gpio)
        except Exception:
            pass

    def _trigger(self):
        self._free()
        try:
            lgpio.gpio_claim_output(self._chip, self._gpio, 0)
        except Exception:
            self._free()  # leave the line released so the next retry starts clean
            raise
        time.sleep(0.001)
        self._bits = 0
        self._code = 0
        lgpio.gpio_claim_alert(self._chip, self._gpio, lgpio.RISING_EDGE)

    def read(self):
        self._new_data = False
        self._status = DHT_TIMEOUT
        self._trigger()
        for _ in range(20):
            time.sleep(0.05)
            if self._new_data:
                break
        return self._status, self._temperature, self._humidity


def stream(sensor, interval):
    while True:
        try:
            status, temperature_c, humidity_percent = sensor.read()
            label = STATUS_LABELS.get(status, str(status))
            if status == DHT_GOOD:
                print(f"{label},{temperature_c:.1f},{humidity_percent:.1f}", flush=True)
            else:
                print(f"{label},,", flush=True)
        except Exception as exc:  # a transient GPIO conflict at boot must not kill the helper
            print("ERROR,,", flush=True)
            print(str(exc), file=sys.stderr)
        time.sleep(max(0.5, interval))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpio", type=int, default=4)
    parser.add_argument("--gpiochip", type=int, default=0)
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()

    chip = lgpio.gpiochip_open(args.gpiochip)
    sensor = DHTSensor(chip, args.gpio, model=DHTXX)

    try:
        stream(sensor, args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        sensor.cancel()
        lgpio.gpiochip_close(chip)


def _selfcheck():
    import contextlib
    import io

    class Fake:
        n = 0

        def read(self):
            Fake.n += 1
            if Fake.n > 2:
                raise KeyboardInterrupt
            raise lgpio.error("GPIO busy")

    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        try:
            stream(Fake(), 0.0)
        except KeyboardInterrupt:
            pass
    assert out.getvalue().count("ERROR,,") == 2, out.getvalue()
    print("selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
        sys.exit(0)
    try:
        main()
    except Exception as exc:
        print(f"ERROR,,", flush=True)
        print(str(exc), file=sys.stderr)
        raise
