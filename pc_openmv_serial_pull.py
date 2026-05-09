from __future__ import annotations

import argparse
import ast
import base64
import time
from datetime import datetime
from pathlib import Path

import serial


PROMPT = b">>> "
ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull a small batch of JPG captures from OpenMV over the serial REPL."
    )
    parser.add_argument("--port", default="COM17", help="OpenMV serial port")
    parser.add_argument("--baudrate", type=int, default=115200, help="Serial baudrate")
    parser.add_argument("--device-folder", default="pc_frames", help="Folder on the OpenMV filesystem")
    parser.add_argument("--expected", type=int, default=6, help="Expected number of images")
    parser.add_argument(
        "--save-root",
        type=Path,
        default=ROOT / "captures",
        help="Local destination root",
    )
    return parser.parse_args()


def read_until_prompt(ser: serial.Serial, timeout: float = 3.0) -> bytes:
    end = time.time() + timeout
    data = bytearray()
    while time.time() < end:
        chunk = ser.read(ser.in_waiting or 1)
        if chunk:
            data.extend(chunk)
            if data.endswith(PROMPT):
                return bytes(data)
        else:
            time.sleep(0.02)
    raise TimeoutError("Timed out waiting for REPL prompt")


def run_command(ser: serial.Serial, command: str, timeout: float = 3.0) -> str:
    ser.reset_input_buffer()
    ser.write(command.encode("utf-8") + b"\r\n")
    data = read_until_prompt(ser, timeout=timeout)
    text = data.decode("utf-8", errors="ignore")
    lines = text.splitlines()
    if lines and lines[0].strip() == command.strip():
        lines = lines[1:]
    if lines and lines[-1].strip() == ">>>":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def enter_repl(ser: serial.Serial) -> None:
    ser.write(b"\x03\x03")
    time.sleep(0.3)
    ser.reset_input_buffer()
    ser.write(b"\r\n")
    read_until_prompt(ser, timeout=4.0)


def list_files(ser: serial.Serial, device_folder: str) -> list[str]:
    out = run_command(ser, f"import os; print(os.listdir('{device_folder}'))", timeout=3.0)
    names = ast.literal_eval(out.splitlines()[-1])
    return [name for name in names if name.lower().endswith(".jpg")]


def pull_file(ser: serial.Serial, device_folder: str, filename: str) -> bytes:
    marker = "__EOF__"
    script = (
        "import ubinascii\n"
        f"f=open('{device_folder}/{filename}','rb')\n"
        "while True:\n"
        "    chunk=f.read(384)\n"
        "    if not chunk:\n"
        "        break\n"
        "    print(ubinascii.b2a_base64(chunk).decode(), end='')\n"
        "f.close()\n"
        f"print('{marker}')\n"
    )
    command = f"exec({script!r})"
    ser.reset_input_buffer()
    ser.write(command.encode("utf-8") + b"\r\n")

    end = time.time() + 20.0
    data = bytearray()
    while time.time() < end:
        chunk = ser.read(ser.in_waiting or 1)
        if chunk:
            data.extend(chunk)
            if marker.encode("utf-8") in data and data.endswith(PROMPT):
                break
        else:
            time.sleep(0.02)
    else:
        raise TimeoutError(f"Timed out pulling {filename}")

    text = data.decode("utf-8", errors="ignore")
    text = text.replace(command + "\r\n", "", 1)
    text = text.rsplit(marker, 1)[0]
    text = text.replace(">>> ", "").strip()
    payload = "".join(line.strip() for line in text.splitlines() if line.strip())
    return base64.b64decode(payload)


def main() -> int:
    args = parse_args()
    session_dir = args.save_root / datetime.now().strftime("openmv_%Y%m%d_%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=True)

    with serial.Serial(args.port, args.baudrate, timeout=0.2) as ser:
        enter_repl(ser)
        files = sorted(list_files(ser, args.device_folder))
        if len(files) < args.expected:
            print(f"only found {len(files)} jpg file(s): {files}")
            return 1

        for name in files[: args.expected]:
            payload = pull_file(ser, args.device_folder, name)
            out_path = session_dir / name
            out_path.write_bytes(payload)
            print(f"saved {out_path}")

    print(f"done: {session_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
