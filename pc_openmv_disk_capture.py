from __future__ import annotations

import argparse
import shutil
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy a small batch of OpenMV on-device captures to the local project."
    )
    parser.add_argument("--device-drive", default="E:", help="OpenMV drive letter, e.g. E:")
    parser.add_argument("--device-folder", default="pc_frames", help="Capture folder on the OpenMV drive")
    parser.add_argument("--expected", type=int, default=6, help="Expected number of images")
    parser.add_argument("--timeout", type=float, default=25.0, help="Wait timeout in seconds")
    parser.add_argument(
        "--save-root",
        type=Path,
        default=ROOT / "captures",
        help="Local root folder for copied images",
    )
    return parser.parse_args()


def list_frames(device_dir: Path) -> list[Path]:
    return sorted(device_dir.glob("*.jpg"))


def main() -> int:
    args = parse_args()
    device_root = Path(args.device_drive + "\\")
    device_dir = device_root / args.device_folder
    deadline = time.time() + args.timeout

    print(f"waiting for device images in {device_dir} ...")
    frames: list[Path] = []
    while time.time() < deadline:
        if device_dir.exists():
            frames = list_frames(device_dir)
            if len(frames) >= args.expected:
                break
        time.sleep(0.5)

    if len(frames) < args.expected:
        print(f"only found {len(frames)} frame(s) in {device_dir}")
        return 1

    session_dir = args.save_root / datetime.now().strftime("openmv_%Y%m%d_%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=True)

    for src in frames[: args.expected]:
        dst = session_dir / src.name
        shutil.copy2(src, dst)
        print(f"copied {src} -> {dst}")

    print(f"done: {session_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
