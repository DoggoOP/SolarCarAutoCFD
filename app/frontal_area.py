"""Frontal silhouette area from the car CAD, before adding the CFD farfield."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np


def _length_scale(raw_min: Sequence[float], raw_max: Sequence[float],
                  cad_min: Sequence[float], cad_max: Sequence[float]) -> float:
    """Convert STEP/STL coordinates to Luminary's metre coordinates."""
    raw_dims = np.subtract(raw_max, raw_min)
    cad_dims = np.subtract(cad_max, cad_min)
    if np.any(raw_dims <= 0) or np.any(cad_dims <= 0):
        raise ValueError("CAD geometry has a zero-size axis")
    ratios = cad_dims / raw_dims
    scale = float(np.median(ratios))
    if not math.isfinite(scale) or scale <= 0 or np.max(np.abs(ratios / scale - 1)) > 0.05:
        raise ValueError(f"Local CAD and Luminary geometry bounds disagree: scale ratios {ratios}")
    return scale


def _silhouette_area(points: np.ndarray, triangles: np.ndarray,
                     pixel_size_m: float = 0.001) -> float:
    """Rasterize the union of YZ-projected triangles; overlapping faces count once."""
    from PIL import Image, ImageDraw

    projected = np.asarray(points, dtype=float)[:, 1:3]
    if not np.all(np.isfinite(projected)):
        raise ValueError("CAD mesh contains non-finite vertices")
    minimum = projected.min(axis=0)
    span = projected.max(axis=0) - minimum
    if np.any(span <= 0):
        raise ValueError("CAD mesh has no frontal extent")
    # Cap memory for unusually large CAD while preserving 1 mm pixels for cars.
    pitch = max(pixel_size_m, math.sqrt(float(np.prod(span)) / 4_000_000))
    width, height = (np.ceil(span / pitch).astype(int) + 3).tolist()
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    pixels = (projected - minimum) / pitch + 1
    for triangle in triangles:
        corners = pixels[triangle]
        # Side faces parallel to the viewing direction have zero projection.
        edge_a, edge_b = corners[1] - corners[0], corners[2] - corners[0]
        cross = edge_a[0] * edge_b[1] - edge_a[1] * edge_b[0]
        if abs(cross) < 1e-10:
            continue
        draw.polygon([tuple(corner) for corner in corners], fill=1)
    area = float(np.count_nonzero(np.asarray(image))) * pitch * pitch
    if area <= 0:
        raise ValueError("CAD mesh has no projected frontal surface")
    return area


def projected_frontal_area(cad_path: Path,
                           cad_min: Sequence[float], cad_max: Sequence[float]) -> float:
    """Compute the union silhouette from uploaded STEP/STL/OBJ geometry in m²."""
    import meshio

    suffix = cad_path.suffix.lower()
    with tempfile.TemporaryDirectory(prefix="autocfd-frontal-") as temp_dir:
        if suffix in {".step", ".stp"}:
            mesh_path = Path(temp_dir) / "car.stl"
            bounds_path = Path(temp_dir) / "bounds.json"
            result = subprocess.run(
                [sys.executable, str(Path(__file__).with_name("step_tessellate.py")), str(cad_path),
                 str(mesh_path), str(bounds_path)],
                capture_output=True, text=True, timeout=180, check=False,
            )
            if result.returncode:
                detail = (result.stderr or result.stdout).strip()[-1000:]
                raise RuntimeError(f"STEP tessellation failed: {detail}")
            bounds = json.loads(bounds_path.read_text())
            raw_min, raw_max = bounds["min"], bounds["max"]
        elif suffix in {".stl", ".obj"}:
            mesh_path = cad_path
            raw_min = raw_max = None
        else:
            raise ValueError(f"Cannot project unsupported CAD format {suffix!r}")

        mesh = meshio.read(mesh_path)
        triangles = [block.data for block in mesh.cells if block.type == "triangle"]
        if not triangles:
            raise ValueError("CAD tessellation contains no surface triangles")
        points = np.asarray(mesh.points, dtype=float)
        if points.shape[1] != 3:
            raise ValueError("CAD mesh must have three-dimensional vertices")
        if raw_min is None:
            raw_min, raw_max = points.min(axis=0), points.max(axis=0)
        scale = _length_scale(raw_min, raw_max, cad_min, cad_max)
        area = _silhouette_area(points * scale, np.concatenate(triangles))
        bbox_area = abs(cad_max[1] - cad_min[1]) * abs(cad_max[2] - cad_min[2])
        if area > bbox_area * 1.02:
            raise ValueError("Projected area exceeds the Luminary bounding box; check CAD units")
        return area
