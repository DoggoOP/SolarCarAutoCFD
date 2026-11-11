"""CAD surface area used as a wetted-area estimate for CFD results."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np

from .frontal_area import _length_scale


def _triangles_area(points: np.ndarray, triangles: np.ndarray) -> float:
    corners = points[triangles]
    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    return float(np.linalg.norm(cross, axis=1).sum() / 2)


def cad_wetted_area(
    cad_path: Path, cad_min: Sequence[float], cad_max: Sequence[float]
) -> float:
    """Measure CAD boundary area in m², using Luminary bounds for unit scaling.

    STEP uses exact OpenCASCADE face areas. STL/OBJ use their surface triangles.
    This includes any enclosed cavity or duplicate surface present in the CAD.
    """
    suffix = cad_path.suffix.lower()
    if suffix in {".step", ".stp"}:
        with tempfile.TemporaryDirectory(prefix="autocfd-wetted-") as temp_dir:
            area_path = Path(temp_dir) / "area.json"
            result = subprocess.run(
                [sys.executable, str(Path(__file__).with_name("step_surface_area.py")),
                 str(cad_path), str(area_path)],
                capture_output=True, text=True, timeout=180, check=False,
            )
            if result.returncode:
                detail = (result.stderr or result.stdout).strip()[-1000:]
                raise RuntimeError(f"STEP surface measurement failed: {detail}")
            measured = json.loads(area_path.read_text())
            scale = _length_scale(measured["min"], measured["max"], cad_min, cad_max)
            area = float(measured["area"]) * scale * scale
    elif suffix in {".stl", ".obj"}:
        import meshio

        mesh = meshio.read(cad_path)
        points = np.asarray(mesh.points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
            raise ValueError("CAD mesh must contain finite three-dimensional vertices")
        scale = _length_scale(points.min(axis=0), points.max(axis=0), cad_min, cad_max)
        triangles = []
        for block in mesh.cells:
            if block.type == "triangle":
                triangles.append(block.data)
            elif block.type == "quad":
                triangles.extend((block.data[:, [0, 1, 2]], block.data[:, [0, 2, 3]]))
        if not triangles:
            raise ValueError("CAD mesh contains no surface triangles")
        area = _triangles_area(points * scale, np.concatenate(triangles))
    else:
        raise ValueError(f"Cannot measure unsupported CAD format {suffix!r}")

    if not math.isfinite(area) or area <= 0:
        raise ValueError("CAD has no positive surface area")
    return area
