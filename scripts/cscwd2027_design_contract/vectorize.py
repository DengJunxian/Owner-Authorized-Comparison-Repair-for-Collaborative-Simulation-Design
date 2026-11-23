"""Presentation-only vector redraw; retain the exact original frozen plot.

Only the matrix rendering primitive changes, from imshow to pcolormesh.
All inputs, counts, dimensions, axes, grayscale and labels come from plot.py.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any
from unittest.mock import patch

import matplotlib.axes  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]

from scripts.cscwd2027_design_contract import plot

NATIVE_TEXT = matplotlib.axes.Axes.text


def legible_text(axis: Any, *args: Any, **kwargs: Any) -> Any:
    """Raise count annotations to eight points without changing their content."""
    if "fontsize" in kwargs:
        kwargs["fontsize"] = max(8, kwargs["fontsize"])
    return NATIVE_TEXT(axis, *args, **kwargs)


def vector_matrix(axis: Any, data: Any, **kwargs: Any) -> Any:
    """Keep imshow's upper origin and half-cell bounds without a raster image."""
    height, width = data.shape
    mesh = axis.pcolormesh(
        np.arange(width + 1) - 0.5,
        np.arange(height + 1) - 0.5,
        data,
        vmin=kwargs["vmin"],
        vmax=kwargs["vmax"],
        cmap=kwargs["cmap"],
        shading="flat",
        rasterized=False,
        antialiased=False,
        linewidth=0,
    )
    axis.set_xlim(-0.5, width - 0.5)
    axis.set_ylim(height - 0.5, -0.5)
    axis.set_aspect(kwargs["aspect"])
    return mesh


def main() -> None:
    """Create separate final vector outputs, keeping frozen outputs unchanged."""
    folder = plot.ROOT / "paper/cscwd2027_design_contract/generated"
    originals = [folder / f"comparison_handoff.{s}" for s in ("pdf", "svg")]
    originals.append(folder / "figure_manifest_design.json")
    if not all(p.exists() for p in originals):
        raise RuntimeError("run the original frozen plot before vectorization")
    backup = {p: p.with_name(p.stem + "_frozen_plot" + p.suffix) for p in originals}
    for path, saved in backup.items():
        if saved.exists() and plot.digest(saved) != plot.digest(path):
            raise RuntimeError("retained frozen plot differs before redraw")
        if not saved.exists():
            shutil.copyfile(path, saved)
    outputs = []
    try:
        with (
            patch.object(matplotlib.axes.Axes, "imshow", vector_matrix),
            patch.object(matplotlib.axes.Axes, "text", legible_text),
        ):
            plot.main()
        for path in originals[:2]:
            target = path.with_name(path.stem + "_vector" + path.suffix)
            shutil.copyfile(path, target)
            outputs.append(target)
    finally:
        for path, saved in backup.items():
            shutil.copyfile(saved, path)
    value = {
        "label": "post-seal presentation-only vector rendering",
        "numeric_and_design_changes": False,
        "presentation_changes": "vector matrix and minimum eight-point count labels",
        "source": {
            str(Path(__file__).relative_to(plot.ROOT)): plot.digest(Path(__file__)),
            str(Path(plot.__file__).relative_to(plot.ROOT)): plot.digest(
                Path(plot.__file__)
            ),
        },
        "frozen_figure_manifest": plot.digest(originals[-1]),
        "retained_frozen_plot": {
            str(p.relative_to(plot.ROOT)): plot.digest(p) for p in backup.values()
        },
        "outputs": {str(p.relative_to(plot.ROOT)): plot.digest(p) for p in outputs},
    }
    (folder / "figure_manifest_vector.json").write_text(
        json.dumps(value, indent=2) + "\n"
    )
    print(json.dumps({"vector_outputs": len(outputs)}))


if __name__ == "__main__":
    main()
