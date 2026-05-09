from __future__ import annotations

import argparse
import time
from pathlib import Path

import serial

from pc_openmv_serial_pull import enter_repl, list_files, pull_file


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trigger OpenMV main.py capture, then pull the JPG batch back to the PC."
    )
    parser.add_argument("--port", default="COM17", help="OpenMV serial port")
    parser.add_argument("--baudrate", type=int, default=115200, help="Serial baudrate")
    parser.add_argument("--device-folder", default="pc_frames", help="Folder on the OpenMV filesystem")
    parser.add_argument("--expected", type=int, default=1, help="Expected number of images")
    parser.add_argument(
        "--save-root",
        type=Path,
        default=ROOT / "captures",
        help="Local destination root",
    )
    return parser.parse_args()


def wait_for_capture_done(ser: serial.Serial, timeout: float = 20.0) -> str:
    marker = "capture_done"
    end = time.time() + timeout
    data = bytearray()
    while time.time() < end:
        chunk = ser.read(ser.in_waiting or 1)
        if chunk:
            data.extend(chunk)
            if marker.encode("utf-8") in data:
                return data.decode("utf-8", errors="ignore")
        else:
            time.sleep(0.02)
    raise TimeoutError("Timed out waiting for OpenMV capture to finish")


def main() -> int:
    args = parse_args()
    args.save_root.mkdir(parents=True, exist_ok=True)

    with serial.Serial(args.port, args.baudrate, timeout=0.2) as ser:
        enter_repl(ser)
        ser.reset_input_buffer()
        ser.write(
            b"import sys\r\n"
            b"sys.modules.pop('main', None)\r\n"
            b"import main\r\n"
        )
        output = wait_for_capture_done(ser)
        print(output.strip())

        enter_repl(ser)
        files = sorted(list_files(ser, args.device_folder))
        if len(files) < args.expected:
            print(f"only found {len(files)} jpg file(s): {files}")
            return 1

        from datetime import datetime

        session_dir = args.save_root / datetime.now().strftime("openmv_%Y%m%d_%H%M%S")
        session_dir.mkdir(parents=True, exist_ok=True)

        for name in files[: args.expected]:
            payload = pull_file(ser, args.device_folder, name)
            out_path = session_dir / name
            out_path.write_bytes(payload)
            print(f"saved {out_path}")

    print(f"done: {session_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
