from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
CALIBRATION_JSON = ROOT / "calibrations" / "dice_homography_20pt_quadratic_corrected.json"
SEMI_AUTO_CSV = ROOT / "coords" / "semi_auto_calibration_points.csv"
OUT_DIR = ROOT / "results" / "calibration_model_validation"


@dataclass
class Dataset:
    pixel: np.ndarray
    world: np.ndarray
    group: np.ndarray
    source: np.ndarray


def load_points() -> Dataset:
    pixels: list[list[float]] = []
    worlds: list[list[float]] = []
    groups: list[str] = []
    sources: list[str] = []

    original = json.loads(CALIBRATION_JSON.read_text(encoding="utf-8"))
    for row in original["points"]:
        pixels.append([float(row["pixel"][0]), float(row["pixel"][1])])
        worlds.append([float(row["world"][0]), float(row["world"][1])])
        groups.append("original20")
        sources.append("original20")

    with SEMI_AUTO_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if not row.get("matched_pixel_x") or not row.get("matched_pixel_y"):
                continue
            pixels.append([float(row["matched_pixel_x"]), float(row["matched_pixel_y"])])
            worlds.append([float(row["world_x"]), float(row["world_y"])])
            groups.append(str(row["session_id"]))
            sources.append("semi_auto")

    return Dataset(
        pixel=np.asarray(pixels, dtype=np.float64),
        world=np.asarray(worlds, dtype=np.float64),
        group=np.asarray(groups, dtype=object),
        source=np.asarray(sources, dtype=object),
    )


def homography_fit(pixel: np.ndarray, world: np.ndarray) -> np.ndarray:
    h, _ = cv2.findHomography(pixel.astype(np.float64), world.astype(np.float64), method=0)
    if h is None:
        raise RuntimeError("cv2.findHomography failed")
    return h.astype(np.float64)


def apply_homography(h: np.ndarray, pixel: np.ndarray) -> np.ndarray:
    points = np.asarray(pixel, dtype=np.float64).reshape(-1, 2)
    homo = np.c_[points, np.ones(len(points))]
    out = (h @ homo.T).T
    return out[:, :2] / out[:, 2:3]


def poly_features(world_xy: np.ndarray, degree: int) -> np.ndarray:
    x = world_xy[:, 0]
    y = world_xy[:, 1]
    cols = [np.ones(len(world_xy)), x, y]
    if degree >= 2:
        cols.extend([x * x, x * y, y * y])
    if degree >= 3:
        cols.extend([x**3, x * x * y, x * y * y, y**3])
    return np.column_stack(cols)


def ridge_fit(features: np.ndarray, target: np.ndarray, lam: float) -> np.ndarray:
    xtx = features.T @ features
    reg = np.eye(xtx.shape[0]) * lam
    reg[0, 0] = 0.0
    return np.linalg.solve(xtx + reg, features.T @ target)


def median_pairwise_distance(points: np.ndarray) -> float:
    if len(points) < 2:
        return 1.0
    diffs = points[:, None, :] - points[None, :, :]
    dist = np.sqrt(np.sum(diffs * diffs, axis=2))
    upper = dist[np.triu_indices(len(points), k=1)]
    return float(np.median(upper[upper > 0]))


def gaussian_kernel(a: np.ndarray, b: np.ndarray, sigma: float) -> np.ndarray:
    diff = a[:, None, :] - b[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    return np.exp(-0.5 * d2 / (sigma * sigma))


def tps_kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = a[:, None, :] - b[None, :, :]
    r2 = np.sum(diff * diff, axis=2)
    out = np.zeros_like(r2)
    mask = r2 > 0
    out[mask] = 0.5 * r2[mask] * np.log(r2[mask])
    return out


class Model:
    name: str

    def fit(self, pixel: np.ndarray, world: np.ndarray) -> None:
        raise NotImplementedError

    def predict(self, pixel: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class HomographyModel(Model):
    def __init__(self, name: str = "homography_refit") -> None:
        self.name = name

    def fit(self, pixel: np.ndarray, world: np.ndarray) -> None:
        self.h = homography_fit(pixel, world)

    def predict(self, pixel: np.ndarray) -> np.ndarray:
        return apply_homography(self.h, pixel)


class FixedHomographyModel(Model):
    def __init__(self) -> None:
        self.name = "existing_homography_fixed"
        data = json.loads(CALIBRATION_JSON.read_text(encoding="utf-8"))
        self.h = np.asarray(data["homography_pixel_to_world"], dtype=np.float64)

    def fit(self, pixel: np.ndarray, world: np.ndarray) -> None:
        return None

    def predict(self, pixel: np.ndarray) -> np.ndarray:
        return apply_homography(self.h, pixel)


class HomographyPolynomialResidual(Model):
    def __init__(self, degree: int, lam: float = 1e-6) -> None:
        self.degree = degree
        self.lam = lam
        names = {1: "homography_affine_residual", 2: "homography_quadratic_residual", 3: "homography_cubic_residual"}
        self.name = names[degree]

    def fit(self, pixel: np.ndarray, world: np.ndarray) -> None:
        self.h = homography_fit(pixel, world)
        raw = apply_homography(self.h, pixel)
        self.coeff = ridge_fit(poly_features(raw, self.degree), world - raw, self.lam)

    def predict(self, pixel: np.ndarray) -> np.ndarray:
        raw = apply_homography(self.h, pixel)
        return raw + poly_features(raw, self.degree) @ self.coeff


class HomographyGaussianRBF(Model):
    def __init__(self, sigma_scale: float = 0.55, lam: float = 0.08) -> None:
        self.name = f"homography_gaussian_rbf_s{sigma_scale:g}_l{lam:g}"
        self.sigma_scale = sigma_scale
        self.lam = lam

    def fit(self, pixel: np.ndarray, world: np.ndarray) -> None:
        self.h = homography_fit(pixel, world)
        raw = apply_homography(self.h, pixel)
        self.centers = raw
        self.sigma = max(1.0, median_pairwise_distance(raw) * self.sigma_scale)
        k = gaussian_kernel(raw, self.centers, self.sigma)
        self.weights = np.linalg.solve(k + self.lam * np.eye(len(k)), world - raw)

    def predict(self, pixel: np.ndarray) -> np.ndarray:
        raw = apply_homography(self.h, pixel)
        return raw + gaussian_kernel(raw, self.centers, self.sigma) @ self.weights


class HomographyTPS(Model):
    def __init__(self, lam: float = 0.2) -> None:
        self.name = f"homography_tps_l{lam:g}"
        self.lam = lam

    def fit(self, pixel: np.ndarray, world: np.ndarray) -> None:
        self.h = homography_fit(pixel, world)
        raw = apply_homography(self.h, pixel)
        self.centers = raw
        k = tps_kernel(raw, raw)
        p = np.c_[np.ones(len(raw)), raw]
        top = np.c_[k + self.lam * np.eye(len(raw)), p]
        bottom = np.c_[p.T, np.zeros((3, 3))]
        lhs = np.r_[top, bottom]
        rhs = np.r_[world - raw, np.zeros((3, 2))]
        sol = np.linalg.solve(lhs, rhs)
        self.weights = sol[: len(raw)]
        self.affine = sol[len(raw) :]

    def predict(self, pixel: np.ndarray) -> np.ndarray:
        raw = apply_homography(self.h, pixel)
        p = np.c_[np.ones(len(raw)), raw]
        return raw + tps_kernel(raw, self.centers) @ self.weights + p @ self.affine


def make_group_folds(groups: np.ndarray, fold_count: int = 5) -> list[list[str]]:
    semi_groups = sorted(g for g in set(groups.tolist()) if g != "original20")
    folds = [[] for _ in range(fold_count)]
    for idx, group in enumerate(semi_groups):
        folds[idx % fold_count].append(group)
    return folds


def error_stats(error_xy: np.ndarray) -> dict[str, float]:
    norm = np.sqrt(np.sum(error_xy * error_xy, axis=1))
    return {
        "count": int(len(norm)),
        "mean_mm": float(np.mean(norm)),
        "rmse_mm": float(math.sqrt(np.mean(norm * norm))),
        "median_mm": float(np.median(norm)),
        "p90_mm": float(np.percentile(norm, 90)),
        "p95_mm": float(np.percentile(norm, 95)),
        "max_mm": float(np.max(norm)),
        "bias_x_mm": float(np.mean(error_xy[:, 0])),
        "bias_y_mm": float(np.mean(error_xy[:, 1])),
    }


def evaluate_model(model_factory, data: Dataset, folds: list[list[str]]) -> tuple[dict, list[dict]]:
    all_errs: list[np.ndarray] = []
    fold_rows: list[dict] = []
    for fold_idx, val_groups in enumerate(folds, start=1):
        val_mask = np.isin(data.group, val_groups)
        train_mask = ~val_mask
        # Validate only on semi-auto held-out sessions; original20 remains an anchor in training.
        val_mask &= data.source == "semi_auto"

        model = model_factory()
        model.fit(data.pixel[train_mask], data.world[train_mask])
        pred = model.predict(data.pixel[val_mask])
        err = pred - data.world[val_mask]
        all_errs.append(err)
        row = {"fold": fold_idx, "val_sessions": ",".join(val_groups), **error_stats(err)}
        fold_rows.append(row)

    total_err = np.vstack(all_errs)
    return error_stats(total_err), fold_rows


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_points()
    folds = make_group_folds(data.group, 5)

    factories = [
        lambda: FixedHomographyModel(),
        lambda: HomographyModel(),
        lambda: HomographyPolynomialResidual(1, 1e-6),
        lambda: HomographyPolynomialResidual(2, 1e-4),
        lambda: HomographyPolynomialResidual(3, 1e-2),
        lambda: HomographyGaussianRBF(0.35, 0.20),
        lambda: HomographyGaussianRBF(0.55, 0.08),
        lambda: HomographyGaussianRBF(0.75, 0.05),
        lambda: HomographyTPS(0.2),
        lambda: HomographyTPS(1.0),
    ]

    summary_rows = []
    fold_rows_all = []
    for factory in factories:
        name = factory().name
        summary, fold_rows = evaluate_model(factory, data, folds)
        summary_rows.append({"model": name, **summary})
        for row in fold_rows:
            fold_rows_all.append({"model": name, **row})

    summary_rows.sort(key=lambda row: (row["rmse_mm"], row["mean_mm"], row["max_mm"]))

    summary_csv = OUT_DIR / "model_validation_summary.csv"
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    fold_csv = OUT_DIR / "model_validation_folds.csv"
    with fold_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fold_rows_all[0].keys()))
        writer.writeheader()
        writer.writerows(fold_rows_all)

    report = {
        "dataset": {
            "original20_points": int(np.sum(data.source == "original20")),
            "semi_auto_valid_points": int(np.sum(data.source == "semi_auto")),
            "total_points": int(len(data.pixel)),
            "semi_auto_session_count": len([g for g in set(data.group.tolist()) if g != "original20"]),
        },
        "validation": {
            "method": "5-fold session holdout",
            "note": "Original 20 calibration points are always kept in training; validation uses held-out semi-auto sessions only.",
            "folds": folds,
        },
        "summary": summary_rows,
    }
    report_json = OUT_DIR / "model_validation_report.json"
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report["dataset"], ensure_ascii=False, indent=2))
    print(f"summary: {summary_csv}")
    print(f"folds: {fold_csv}")
    print(f"report: {report_json}")
    print("\nTop models:")
    for row in summary_rows[:8]:
        print(
            f"{row['model']}: mean={row['mean_mm']:.3f} rmse={row['rmse_mm']:.3f} "
            f"p95={row['p95_mm']:.3f} max={row['max_mm']:.3f} "
            f"bias=({row['bias_x_mm']:+.3f},{row['bias_y_mm']:+.3f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
