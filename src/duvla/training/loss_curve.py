"""Dependency-free loss-curve artifacts for long Duvla training runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape


LOSS_ORDER = (
    "flow",
    "direct",
    "instruction",
    "monotonic",
    "gripper",
    "gripper_transition",
)


def append_loss_point(path: str | Path, point: dict[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a") as handle:
        handle.write(json.dumps(point) + "\n")


def read_loss_points(path: str | Path) -> list[dict[str, object]]:
    source = Path(path)
    if not source.is_file():
        return []
    return [json.loads(line) for line in source.read_text().splitlines() if line.strip()]


def truncate_loss_points(path: str | Path, *, max_optimizer_step: int) -> None:
    destination = Path(path)
    points = [
        point
        for point in read_loss_points(destination)
        if int(point["optimizer_step"]) <= max_optimizer_step
    ]
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(point) + "\n" for point in points))
    temporary.replace(destination)


def _loss_value(point: dict[str, object], name: str) -> float:
    losses = point.get("losses")
    if not isinstance(losses, dict) or name not in losses:
        raise ValueError(f"loss point is missing {name}")
    return float(losses[name])


def _loss_names(points: list[dict[str, object]]) -> tuple[str, ...]:
    available: set[str] = set()
    for point in points:
        losses = point.get("losses")
        if not isinstance(losses, dict):
            raise ValueError("loss point must contain a losses mapping")
        available.update(str(name) for name in losses)
    ordered = [name for name in LOSS_ORDER if name in available]
    ordered.extend(sorted(available.difference(ordered)))
    return tuple(ordered)


def _write_csv(
    path: Path,
    points: list[dict[str, object]],
    loss_names: tuple[str, ...],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "optimizer_step",
                "effective_epochs",
                "learning_rate",
                *loss_names,
            ]
        )
        for point in points:
            learning_rate = point.get("learning_rate")
            if learning_rate is None:
                learning_rates = point.get("learning_rates")
                if not isinstance(learning_rates, list) or not learning_rates:
                    raise ValueError("loss point must contain learning_rate(s)")
                learning_rate = learning_rates[0]
            writer.writerow(
                [
                    int(point["optimizer_step"]),
                    float(point["effective_epochs"]),
                    float(learning_rate),
                    *[_loss_value(point, name) for name in loss_names],
                ]
            )
    temporary.replace(path)


def _write_svg(
    path: Path,
    points: list[dict[str, object]],
    *,
    title: str,
    loss_names: tuple[str, ...],
) -> None:
    width = 960
    left, right, top = 88.0, 24.0, 64.0
    panel_height, panel_gap = 170.0, 42.0
    height = int(top + len(loss_names) * (panel_height + panel_gap) + 24)
    plot_width = width - left - right
    steps = [int(point["optimizer_step"]) for point in points]
    x_min, x_max = min(steps), max(steps)
    if x_min == x_max:
        x_max = x_min + 1
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="34" font-family="sans-serif" font-size="22" fill="#111827">{escape(title)}</text>',
    ]
    colors = ("#2563eb", "#7c3aed", "#0891b2", "#d97706", "#059669", "#dc2626")
    for panel, name in enumerate(loss_names):
        color = colors[panel % len(colors)]
        y_top = top + panel * (panel_height + panel_gap)
        values = [_loss_value(point, name) for point in points]
        y_min, y_max = min(values), max(values)
        if y_min == y_max:
            padding = max(abs(y_min) * 0.05, 1e-6)
            y_min -= padding
            y_max += padding

        def x_coord(step: int) -> float:
            return left + (step - x_min) / (x_max - x_min) * plot_width

        def y_coord(value: float) -> float:
            return y_top + panel_height - (value - y_min) / (y_max - y_min) * panel_height

        polyline = " ".join(
            f"{x_coord(step):.2f},{y_coord(value):.2f}"
            for step, value in zip(steps, values, strict=True)
        )
        elements.extend(
            [
                f'<rect x="{left}" y="{y_top}" width="{plot_width}" height="{panel_height}" fill="#f9fafb" stroke="#d1d5db"/>',
                f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2.5"/>',
                f'<text x="16" y="{y_top + 18}" font-family="sans-serif" font-size="14" fill="{color}">{escape(name)}</text>',
                f'<text x="{left - 8}" y="{y_top + 12}" text-anchor="end" font-family="monospace" font-size="11" fill="#4b5563">{y_max:.6g}</text>',
                f'<text x="{left - 8}" y="{y_top + panel_height}" text-anchor="end" font-family="monospace" font-size="11" fill="#4b5563">{y_min:.6g}</text>',
            ]
        )
    elements.extend(
        [
            f'<text x="{left}" y="{height - 18}" font-family="monospace" font-size="12" fill="#4b5563">step {x_min}</text>',
            f'<text x="{width - right}" y="{height - 18}" text-anchor="end" font-family="monospace" font-size="12" fill="#4b5563">step {max(steps)}</text>',
            "</svg>",
        ]
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(elements) + "\n")
    temporary.replace(path)


def write_loss_curve_artifacts(
    source: str | Path,
    *,
    csv_path: str | Path,
    svg_path: str | Path,
    title: str,
) -> None:
    points = read_loss_points(source)
    if not points:
        raise ValueError("cannot render an empty loss curve")
    loss_names = _loss_names(points)
    _write_csv(Path(csv_path), points, loss_names)
    _write_svg(Path(svg_path), points, title=title, loss_names=loss_names)
