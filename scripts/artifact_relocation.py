#!/usr/bin/env python
"""Helpers to relocate default script artifacts into case-specific output dirs."""

from __future__ import annotations

import shutil
from pathlib import Path


def snapshot_default_artifacts() -> dict[str, set[Path]]:
    """Capture existing files in default results/logs roots."""
    roots = {"results": Path("results"), "logs": Path("logs")}
    snapshot: dict[str, set[Path]] = {}
    for name, root in roots.items():
        if root.exists():
            snapshot[name] = {path.resolve() for path in root.rglob("*") if path.is_file()}
        else:
            snapshot[name] = set()
    return snapshot


def relocate_new_default_artifacts(
    *, snapshot: dict[str, set[Path]], output_dir: str
) -> list[tuple[Path, Path]]:
    """Move new files generated in default roots into output_dir."""
    output_root = Path(output_dir).resolve()
    roots = {"results": Path("results"), "logs": Path("logs")}
    moved: list[tuple[Path, Path]] = []

    for name, root in roots.items():
        if not root.exists():
            continue
        root_resolved = root.resolve()
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            source = path.resolve()
            is_experiments_log = name == "logs" and source.name.startswith("experiments-")
            if source in snapshot.get(name, set()) and not is_experiments_log:
                continue
            if source.is_relative_to(output_root):
                continue
            relative = source.relative_to(root_resolved)
            if name == "results" and relative.parts and relative.parts[0].startswith("case"):
                continue
            target = output_root / name / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
            moved.append((source, target))

    return moved
