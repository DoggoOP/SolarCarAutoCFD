"""Measure imported STEP surfaces with OpenCASCADE in an isolated process."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def main() -> None:
    import gmsh

    cad_path, output_path = map(Path, sys.argv[1:3])
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("autocfd_cad_surface_area")
        gmsh.model.occ.importShapes(str(cad_path))
        gmsh.model.occ.synchronize()

        volumes = gmsh.model.getEntities(3)
        if len(volumes) > 1:
            # Use the union of solid bodies so touching/overlapping part faces
            # are not counted as fluid-exposed surface on both sides.
            gmsh.model.occ.fuse(volumes[:1], volumes[1:])
            gmsh.model.occ.synchronize()
            volumes = gmsh.model.getEntities(3)
        all_surfaces = set(gmsh.model.getEntities(2))
        if not all_surfaces:
            raise ValueError("STEP file has no surfaces")

        if volumes:
            outer_surfaces = set()
            solid_surfaces = set()
            for _, volume_tag in volumes:
                _, surface_loops = gmsh.model.occ.getSurfaceLoops(volume_tag)
                if not surface_loops:
                    raise ValueError(f"STEP volume {volume_tag} has no surface shell")

                def loop_extent(face_tags: list[int]) -> float:
                    boxes = [gmsh.model.getBoundingBox(2, abs(int(tag))) for tag in face_tags]
                    minimum = [min(box[axis] for box in boxes) for axis in range(3)]
                    maximum = [max(box[axis + 3] for box in boxes) for axis in range(3)]
                    return math.prod(maximum[axis] - minimum[axis] for axis in range(3))

                # The outer shell encloses the other loops. Sealed internal
                # cavities have separate, smaller loops and are not wind-wetted.
                outer_loop = max(surface_loops, key=loop_extent)
                outer_surfaces.update((2, abs(int(tag))) for tag in outer_loop)
                for loop in surface_loops:
                    solid_surfaces.update((2, abs(int(tag))) for tag in loop)

            # A STEP assembly can also contain standalone surface bodies.
            selected_surfaces = outer_surfaces | (all_surfaces - solid_surfaces)
        else:
            selected_surfaces = all_surfaces

        raw_area = sum(gmsh.model.occ.getMass(2, tag) for _, tag in selected_surfaces)
        if not math.isfinite(raw_area) or raw_area <= 0:
            raise ValueError("STEP file has no positive surface area")
        bbox = gmsh.model.getBoundingBox(-1, -1)
        output_path.write_text(json.dumps({
            "min": bbox[:3],
            "max": bbox[3:],
            "area": raw_area,
            "surface_count": len(selected_surfaces),
        }))
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    main()
