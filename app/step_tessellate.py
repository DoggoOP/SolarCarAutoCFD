"""Run Gmsh in a separate process to triangulate STEP CAD for frontal area."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    import gmsh

    cad_path, mesh_path, bounds_path = map(Path, sys.argv[1:4])
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("autocfd_frontal_area")
        gmsh.model.occ.importShapes(str(cad_path))
        gmsh.model.occ.synchronize()
        if not gmsh.model.getEntities(2):
            raise ValueError("STEP file has no surfaces")
        bbox = gmsh.model.getBoundingBox(-1, -1)
        span = max(bbox[3 + axis] - bbox[axis] for axis in range(3))
        if span <= 0:
            raise ValueError("STEP file has an invalid bounding box")
        gmsh.option.setNumber("Mesh.MeshSizeMin", span / 1000)
        gmsh.option.setNumber("Mesh.MeshSizeMax", span / 120)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.SaveAll", 1)
        gmsh.option.setNumber("Mesh.Binary", 1)
        gmsh.model.mesh.generate(2)
        gmsh.write(str(mesh_path))
        bounds_path.write_text(json.dumps({"min": bbox[:3], "max": bbox[3:]}))
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    main()
