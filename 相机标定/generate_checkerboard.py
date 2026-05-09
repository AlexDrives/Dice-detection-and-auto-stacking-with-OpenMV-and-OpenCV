import argparse
from pathlib import Path

import cv2
import numpy as np


def mm_to_px(mm: float, dpi: int) -> int:
    return int(round(mm / 25.4 * dpi))


def build_checkerboard(inner_cols: int, inner_rows: int, square_mm: float, dpi: int, margin_mm: float):
    squares_x = inner_cols + 1
    squares_y = inner_rows + 1
    square_px = mm_to_px(square_mm, dpi)
    margin_px = mm_to_px(margin_mm, dpi)

    board_w = squares_x * square_px
    board_h = squares_y * square_px
    img_w = board_w + 2 * margin_px
    img_h = board_h + 2 * margin_px

    image = np.full((img_h, img_w), 255, dtype=np.uint8)
    for y in range(squares_y):
        for x in range(squares_x):
            if (x + y) % 2 == 0:
                x0 = margin_px + x * square_px
                y0 = margin_px + y * square_px
                image[y0 : y0 + square_px, x0 : x0 + square_px] = 0
    return image, squares_x, squares_y


def write_svg(path: Path, inner_cols: int, inner_rows: int, square_mm: float, margin_mm: float) -> None:
    squares_x = inner_cols + 1
    squares_y = inner_rows + 1
    width_mm = squares_x * square_mm + 2 * margin_mm
    height_mm = squares_y * square_mm + 2 * margin_mm

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_mm}mm" height="{height_mm}mm" '
        f'viewBox="0 0 {width_mm} {height_mm}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    for y in range(squares_y):
        for x in range(squares_x):
            if (x + y) % 2 == 0:
                px = margin_mm + x * square_mm
                py = margin_mm + y * square_mm
                parts.append(
                    f'<rect x="{px}" y="{py}" width="{square_mm}" height="{square_mm}" fill="black"/>'
                )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a printable OpenCV checkerboard.")
    parser.add_argument("--inner-cols", type=int, default=9, help="number of inner corners along x")
    parser.add_argument("--inner-rows", type=int, default=6, help="number of inner corners along y")
    parser.add_argument("--square-mm", type=float, default=5.0, help="square side length in mm")
    parser.add_argument("--margin-mm", type=float, default=10.0, help="white margin around board in mm")
    parser.add_argument("--dpi", type=int, default=300, help="PNG rendering DPI")
    parser.add_argument("--out-dir", default="boards", help="output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    name = f"checkerboard_{args.inner_cols}x{args.inner_rows}_inner_{args.square_mm:g}mm"
    png_path = out_dir / f"{name}.png"
    svg_path = out_dir / f"{name}.svg"

    image, squares_x, squares_y = build_checkerboard(
        args.inner_cols, args.inner_rows, args.square_mm, args.dpi, args.margin_mm
    )
    cv2.imencode(".png", image)[1].tofile(str(png_path))
    write_svg(svg_path, args.inner_cols, args.inner_rows, args.square_mm, args.margin_mm)

    print(f"Saved PNG: {png_path}")
    print(f"Saved SVG: {svg_path}")
    print(f"Inner corners: {args.inner_cols} x {args.inner_rows}")
    print(f"Printed squares: {squares_x} x {squares_y}")
    print(f"Square size: {args.square_mm} mm")


if __name__ == "__main__":
    main()
