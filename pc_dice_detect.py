from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from camera_undistort import load_camera_calibration, undistort_image


ROOT = Path(__file__).resolve().parent
PATCH_SIZE = 42
BODY_THRESHOLD = 170
MIN_DICE_AREA = 100
MAX_DICE_AREA = 3000
PEAK_THRESHOLD = 0.18
PEAK_NMS_DIST2 = 20
MAX_PEAKS = 8
MIN_DICE_W = 8
MIN_DICE_H = 8
MAX_DICE_W = 90
MAX_DICE_H = 90
CENTER_CORRECT_DX0 = -0.8
CENTER_CORRECT_DX_PER_X = -0.003
CENTER_CORRECT_DY0 = -1.25
CENTER_CORRECT_DY_PER_Y = -0.006
CENTER_CORRECT_REF_X = 160.0
CENTER_CORRECT_REF_Y = 120.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect dice values and pip-center pixel coordinates from local OpenMV capture images."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input image file or directory of JPG/PNG images.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=ROOT / "results" / "dice_detect",
        help="Directory for annotated outputs and JSON summaries.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show annotated windows while processing.",
    )
    parser.add_argument(
        "--body-threshold",
        type=int,
        default=BODY_THRESHOLD,
        help="Fixed grayscale threshold for direct dice-body binarization.",
    )
    parser.add_argument(
        "--save-binary",
        action="store_true",
        help="Save the fixed-threshold binary dice-body mask for debugging.",
    )
    parser.add_argument(
        "--intrinsics",
        type=Path,
        default=None,
        help="Optional camera_intrinsics.json/yaml. If provided, images are undistorted before detection.",
    )
    parser.add_argument(
        "--undistort-alpha",
        type=float,
        default=1.0,
        help="OpenCV optimal new camera matrix alpha. 1 keeps full FOV, 0 crops invalid borders.",
    )
    return parser.parse_args()


def collect_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    return sorted([p for p in path.iterdir() if p.suffix.lower() in exts])


def dice_body_mask(image: np.ndarray, threshold: int) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return mask


def detect_dice_centers(image: np.ndarray, threshold: int) -> list[tuple[float, float]]:
    mask = dice_body_mask(image, threshold)
    num, _, stats, cents = cv2.connectedComponentsWithStats(mask)

    centers: list[tuple[float, float]] = []
    for idx in range(1, num):
        _, _, _, _, area = stats[idx]
        if MIN_DICE_AREA <= area <= MAX_DICE_AREA:
            centers.append((float(cents[idx][0]), float(cents[idx][1])))

    centers.sort(key=lambda p: (p[1], p[0]))
    return centers


def looks_like_dice_component(w: int, h: int, area: int) -> bool:
    if not (MIN_DICE_AREA <= area <= MAX_DICE_AREA):
        return False
    if not (MIN_DICE_W <= w <= MAX_DICE_W and MIN_DICE_H <= h <= MAX_DICE_H):
        return False
    aspect = w / max(h, 1)
    if not (0.45 <= aspect <= 2.2):
        return False
    fill_ratio = area / max(w * h, 1)
    return 0.25 <= fill_ratio <= 0.95


def detect_dice_regions(image: np.ndarray, threshold: int) -> list[dict[str, object]]:
    mask = dice_body_mask(image, threshold)
    num, labels, stats, cents = cv2.connectedComponentsWithStats(mask)

    regions: list[dict[str, object]] = []
    for idx in range(1, num):
        x, y, w, h, area = stats[idx]
        if not looks_like_dice_component(int(w), int(h), int(area)):
            continue
        component = (labels[y : y + h, x : x + w] == idx).astype(np.uint8) * 255
        regions.append(
            {
                "center": (float(cents[idx][0]), float(cents[idx][1])),
                "bbox": (int(x), int(y), int(x + w), int(y + h)),
                "component": component,
            }
        )

    regions.sort(key=lambda r: (r["center"][1], r["center"][0]))
    return regions


def crop_patch(image: np.ndarray, center: tuple[float, float], size: int = PATCH_SIZE) -> tuple[np.ndarray, int, int]:
    cx, cy = [int(round(v)) for v in center]
    half = size // 2
    x0 = max(0, cx - half)
    y0 = max(0, cy - half)
    x1 = min(image.shape[1], x0 + size)
    y1 = min(image.shape[0], y0 + size)

    x0 = max(0, x1 - size)
    y0 = max(0, y1 - size)
    patch = image[y0:y1, x0:x1]
    return patch, x0, y0


def local_response_map(patch: np.ndarray, patch_mask: np.ndarray | None = None) -> np.ndarray:
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

    bg = cv2.GaussianBlur(gray, (17, 17), 0)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    dark = cv2.subtract(bg, blur).astype(np.float32) / 255.0

    red = cv2.bitwise_or(
        cv2.inRange(hsv, (0, 70, 40), (12, 255, 255)),
        cv2.inRange(hsv, (160, 70, 40), (179, 255, 255)),
    ).astype(np.float32) / 255.0
    blue = cv2.inRange(hsv, (90, 40, 20), (145, 255, 255)).astype(np.float32) / 255.0

    resp = np.maximum(dark * 1.25, np.maximum(red, blue))

    h, w = resp.shape
    yy, xx = np.mgrid[0:h, 0:w]
    border = np.minimum.reduce([xx, yy, w - 1 - xx, h - 1 - yy]).astype(np.float32)
    resp *= np.clip((border - 2) / 5, 0, 1)
    return cv2.GaussianBlur(resp, (0, 0), 1.0)


def extract_peaks(resp: np.ndarray) -> list[tuple[float, int, int]]:
    dilated = cv2.dilate(resp, np.ones((5, 5), np.float32))
    ys, xs = np.where((resp >= dilated - 1e-6) & (resp > PEAK_THRESHOLD))
    raw = sorted(
        [(float(resp[y, x]), int(x), int(y)) for x, y in zip(xs, ys)],
        reverse=True,
    )

    selected: list[tuple[float, int, int]] = []
    for score, x, y in raw:
        if all((x - px) ** 2 + (y - py) ** 2 >= PEAK_NMS_DIST2 for _, px, py in selected):
            selected.append((score, x, y))
        if len(selected) >= MAX_PEAKS:
            break
    return selected


def color_features(patch: np.ndarray, restrict_to_body: bool = False) -> dict[str, object]:
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    red_mask = cv2.bitwise_or(
        cv2.inRange(hsv, (0, 70, 40), (12, 255, 255)),
        cv2.inRange(hsv, (160, 70, 40), (179, 255, 255)),
    )
    blue_mask = cv2.inRange(hsv, (85, 25, 10), (150, 255, 255))

    if restrict_to_body:
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        _, body_mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        num_body, body_labels, body_stats, _ = cv2.connectedComponentsWithStats(body_mask)
        body_component = np.zeros_like(body_mask)
        patch_cx = patch.shape[1] // 2
        patch_cy = patch.shape[0] // 2
        picked_label = 0
        if 0 <= patch_cy < body_labels.shape[0] and 0 <= patch_cx < body_labels.shape[1]:
            picked_label = int(body_labels[patch_cy, patch_cx])
        if picked_label == 0:
            best_label = 0
            best_dist = None
            for idx in range(1, num_body):
                x, y, w, h, area = body_stats[idx]
                if area < 80:
                    continue
                cx = x + (w / 2.0)
                cy = y + (h / 2.0)
                dist = (cx - patch_cx) ** 2 + (cy - patch_cy) ** 2
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_label = idx
            picked_label = best_label
        if picked_label > 0:
            body_component[body_labels == picked_label] = 255
            body_component = cv2.erode(body_component, np.ones((3, 3), np.uint8), iterations=1)
            red_mask = cv2.bitwise_and(red_mask, body_component)
            blue_mask = cv2.bitwise_and(blue_mask, body_component)

    kernel = np.ones((3, 3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)

    def blob_stats(mask: np.ndarray) -> tuple[list[int], list[tuple[float, float]]]:
        num, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        areas: list[int] = []
        centers: list[tuple[float, float]] = []
        for idx in range(1, num):
            x, y, w, h, area = stats[idx]
            if 6 <= area <= 120 and 3 <= w <= 12 and 3 <= h <= 12:
                roi = mask[y : y + h, x : x + w]
                moments = cv2.moments(roi, binaryImage=True)
                if moments["m00"] > 0:
                    cx = x + (moments["m10"] / moments["m00"])
                    cy = y + (moments["m01"] / moments["m00"])
                else:
                    cx = x + (w / 2.0)
                    cy = y + (h / 2.0)
                areas.append(int(area))
                centers.append((float(cx), float(cy)))
        return areas, centers

    red_areas, red_centers = blob_stats(red_mask)
    blue_areas, blue_centers = blob_stats(blue_mask)

    return {
        "red_sum": int(red_mask.sum()),
        "blue_sum": int(blue_mask.sum()),
        "red_blobs": red_areas,
        "blue_blobs": blue_areas,
        "red_centers": red_centers,
        "blue_centers": blue_centers,
    }


def count_from_peaks(peaks: list[tuple[float, int, int]]) -> int:
    if not peaks:
        return 0

    vals = [p[0] for p in peaks[:6]]
    max_n = min(6, len(vals))
    if max_n == 1:
        return 1

    gaps = [vals[i] - vals[i + 1] for i in range(max_n - 1)]
    best_gap = max(gaps)
    if best_gap > 0.12:
        return gaps.index(best_gap) + 1
    return max_n


def binary_pips_from_region(region: dict[str, object]) -> list[tuple[float, float]]:
    x0, y0, _, _ = region["bbox"]
    component = region["component"]
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(component)
    cv2.drawContours(filled, contours, -1, 255, -1)
    holes = cv2.subtract(filled, component)
    holes = cv2.morphologyEx(holes, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    holes = cv2.erode(holes, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=1)

    num, labels, stats, cents = cv2.connectedComponentsWithStats(holes)
    centers: list[tuple[float, float]] = []
    for idx in range(1, num):
        hx, hy, hw, hh, area = stats[idx]
        if not (3 <= area <= 260 and 1 <= hw <= 30 and 1 <= hh <= 30):
            continue

        centers.append((float(cents[idx][0] + x0), float(cents[idx][1] + y0)))

    return centers


def center_pip_from_odd_pattern(pts: np.ndarray) -> np.ndarray:
    centroid = pts.mean(axis=0)
    distances = np.sum((pts - centroid) ** 2, axis=1)
    return pts[int(np.argmin(distances))]


def six_corner_pips(pts: np.ndarray) -> np.ndarray:
    centroid = pts.mean(axis=0)
    distances = np.sum((pts - centroid) ** 2, axis=1)
    corner_indices = np.argsort(distances)[-4:]
    return pts[corner_indices].astype(np.float32)


def quad_pip_center(pts: np.ndarray) -> np.ndarray:
    return pts.mean(axis=0)


def center_pip_candidate(pts: np.ndarray) -> tuple[bool, np.ndarray | None]:
    centroid = pts.mean(axis=0)
    span = np.ptp(pts, axis=0)
    scale = float(max(span[0], span[1], 1.0))
    distances = np.linalg.norm(pts - centroid, axis=1)
    near = pts[distances < 0.22 * scale]
    if len(near) == 0:
        return False, None
    if len(near) > 1 and float(max(np.ptp(near, axis=0))) > 0.18 * scale:
        return False, None
    return True, near.mean(axis=0).astype(np.float32)


def classify_pip_pattern(pts: np.ndarray) -> tuple[int, np.ndarray]:
    count = len(pts)
    has_center, center_pip = center_pip_candidate(pts)

    if count <= 1:
        return count, pts[0]
    if count == 2:
        return 2, pts.mean(axis=0)
    if count == 3:
        return 3, center_pip if has_center and center_pip is not None else center_pip_from_odd_pattern(pts)
    if count == 4:
        return 4, quad_pip_center(pts)
    if count == 5:
        if has_center and center_pip is not None:
            return 5, center_pip
        return 6, quad_pip_center(six_corner_pips(pts))
    if count >= 6:
        if has_center and center_pip is not None:
            return 5, center_pip
        return 6, quad_pip_center(six_corner_pips(pts))

    return 0, pts.mean(axis=0)


def correct_pip_center(x: float, y: float) -> tuple[float, float]:
    dx = CENTER_CORRECT_DX0 + CENTER_CORRECT_DX_PER_X * (x - CENTER_CORRECT_REF_X)
    dy = CENTER_CORRECT_DY0 + CENTER_CORRECT_DY_PER_Y * (y - CENTER_CORRECT_REF_Y)
    return float(x + dx), float(y + dy)


def pip_value_and_center(region: dict[str, object]) -> tuple[int, tuple[float, float]]:
    pips = binary_pips_from_region(region)
    if not pips:
        return 0, region["center"]

    pts = np.array(pips, dtype=np.float32)
    value, center = classify_pip_pattern(pts)
    return int(value), (float(center[0]), float(center[1]))


def color_from_patch(patch: np.ndarray) -> str:
    features = color_features(patch)
    red_sum = int(features["red_sum"])
    blue_sum = int(features["blue_sum"])
    if red_sum > blue_sum * 1.4 and red_sum > 1000:
        return "r"
    if blue_sum > 1000:
        return "b"
    return "u"


def color_from_value(value: int, fallback: str) -> str:
    if value in (1, 4):
        return "r"
    if value in (2, 3, 5, 6):
        return "b"
    return fallback


def classify_patch(patch: np.ndarray) -> tuple[int, str, list[tuple[float, int, int]]]:
    peaks = extract_peaks(local_response_map(patch))
    colors = color_features(patch)

    red_blobs = len(colors["red_blobs"])
    red_sum = int(colors["red_sum"])
    blue_sum = int(colors["blue_sum"])

    if red_blobs >= 3:
        return 4, "r", peaks
    if red_blobs == 1 and red_sum >= 12000:
        return 1, "r", peaks

    value = count_from_peaks(peaks)
    color = "b" if blue_sum >= red_sum else "u"
    return value, color, peaks


def pip_center_from_peaks(
    peaks: list[tuple[float, int, int]],
    x0: int,
    y0: int,
    fallback_center: tuple[float, float],
    value: int,
    color: str,
    color_centers: list[tuple[float, float]],
) -> tuple[float, float]:
    if color_centers:
        pts = np.array(color_centers, dtype=np.float32)
        if len(color_centers) == 1:
            center = pts[0]
        elif len(color_centers) == 2:
            center = pts.mean(axis=0)
        else:
            rect = cv2.minAreaRect(pts)
            center = np.array(rect[0], dtype=np.float32)
        return float(x0 + center[0]), float(y0 + center[1])

    if not peaks or value <= 0:
        return fallback_center

    top_score = peaks[0][0]
    strong = [
        (score, x, y)
        for score, x, y in peaks
        if score >= max(0.45, top_score * 0.72)
    ]
    if not strong:
        strong = peaks[: min(value, len(peaks))]

    if color == "r":
        chosen = strong[: min(max(value, 1), len(strong))]
    else:
        chosen = strong

    if len(chosen) == 1:
        _, x, y = chosen[0]
        return float(x0 + x), float(y0 + y)

    pts = np.array([[x, y] for _, x, y in chosen], dtype=np.float32)

    if len(chosen) == 2:
        center = pts.mean(axis=0)
    else:
        rect = cv2.minAreaRect(pts)
        center = np.array(rect[0], dtype=np.float32)

    cx = float(x0 + center[0])
    cy = float(y0 + center[1])
    return cx, cy


def annotate(image: np.ndarray, detections: list[dict[str, object]]) -> np.ndarray:
    out = image.copy()
    for det in detections:
        cx = int(round(float(det["die_center_x"])))
        cy = int(round(float(det["die_center_y"])))
        px = int(round(float(det["pip_center_x"])))
        py = int(round(float(det["pip_center_y"])))
        value = int(det["value"])
        color = str(det["color"])

        x0 = int(det.get("bbox_x0", max(0, cx - PATCH_SIZE // 2)))
        y0 = int(det.get("bbox_y0", max(0, cy - PATCH_SIZE // 2)))
        x1 = int(det.get("bbox_x1", min(out.shape[1] - 1, cx + PATCH_SIZE // 2)))
        y1 = int(det.get("bbox_y1", min(out.shape[0] - 1, cy + PATCH_SIZE // 2)))

        box_color = (0, 0, 255) if color == "r" else (255, 0, 0)
        cv2.rectangle(out, (x0, y0), (x1, y1), box_color, 1)
        cv2.drawMarker(out, (px, py), (0, 255, 255), cv2.MARKER_CROSS, 7, 1)
        cv2.putText(
            out,
            f"{value}",
            (x0, max(16, y0 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            box_color,
            1,
            cv2.LINE_AA,
        )
    return out


def process_image(
    path: Path,
    save_dir: Path,
    show: bool,
    body_threshold: int,
    save_binary: bool,
    intrinsics: Path | None = None,
    undistort_alpha: float = 1.0,
) -> dict[str, object]:
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Cannot read image: {path}")

    undistorted_path = None
    calibration_used = None
    if intrinsics is not None:
        camera_matrix, dist_coeffs, _ = load_camera_calibration(intrinsics)
        image, new_matrix = undistort_image(image, camera_matrix, dist_coeffs, undistort_alpha)
        calibration_used = {
            "intrinsics": str(intrinsics),
            "undistort_alpha": float(undistort_alpha),
            "new_camera_matrix": new_matrix.tolist(),
        }

    regions = detect_dice_regions(image, body_threshold)
    detections: list[dict[str, object]] = []

    for region in regions:
        cx, cy = region["center"]
        patch, x0, y0 = crop_patch(image, (cx, cy))
        fallback_color = color_from_patch(patch)
        value, (pip_cx, pip_cy) = pip_value_and_center(region)
        if value <= 0:
            continue
        raw_pip_cx, raw_pip_cy = pip_cx, pip_cy
        pip_cx, pip_cy = correct_pip_center(pip_cx, pip_cy)
        color = color_from_value(value, fallback_color)

        detections.append(
            {
                "value": int(value),
                "color": color,
                "die_center_x": round(cx, 2),
                "die_center_y": round(cy, 2),
                "raw_pip_center_x": round(raw_pip_cx, 2),
                "raw_pip_center_y": round(raw_pip_cy, 2),
                "pip_center_x": round(pip_cx, 2),
                "pip_center_y": round(pip_cy, 2),
                "bbox_x0": int(region["bbox"][0]),
                "bbox_y0": int(region["bbox"][1]),
                "bbox_x1": int(region["bbox"][2]),
                "bbox_y1": int(region["bbox"][3]),
            }
        )

    annotated = annotate(image, detections)
    save_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = save_dir / f"{path.stem}_annotated.png"
    json_path = save_dir / f"{path.stem}.json"
    cv2.imwrite(str(annotated_path), annotated)
    if intrinsics is not None:
        undistorted_path = save_dir / f"{path.stem}_undistorted.png"
        cv2.imwrite(str(undistorted_path), image)
    binary_path = None
    if save_binary:
        binary_path = save_dir / f"{path.stem}_binary_t{body_threshold}.png"
        cv2.imwrite(str(binary_path), dice_body_mask(image, body_threshold))
    json_path.write_text(json.dumps(detections, ensure_ascii=False, indent=2), encoding="utf-8")

    if show:
        cv2.imshow(str(path.name), annotated)
        cv2.waitKey(0)

    print(path.name)
    for det in detections:
        print(
            "  ({value}, {x:.1f}, {y:.1f}, {color})".format(
                value=det["value"],
                x=det["pip_center_x"],
                y=det["pip_center_y"],
                color=det["color"],
            )
        )

    return {
        "image": str(path),
        "undistorted": str(undistorted_path) if undistorted_path else None,
        "calibration": calibration_used,
        "annotated": str(annotated_path),
        "binary": str(binary_path) if binary_path else None,
        "json": str(json_path),
        "detections": detections,
    }


def main() -> int:
    args = parse_args()
    images = collect_images(args.input)
    if not images:
        raise SystemExit(f"No images found in {args.input}")

    summaries = []
    for image_path in images:
        summaries.append(
            process_image(
                image_path,
                args.save_dir,
                args.show,
                args.body_threshold,
                args.save_binary,
                args.intrinsics,
                args.undistort_alpha,
            )
        )

    summary_path = args.save_dir / "summary.json"
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.show:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
