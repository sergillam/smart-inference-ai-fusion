#!/usr/bin/env python
"""Helpers to relocate default script artifacts into case-specific output dirs."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Iterable


def _default_roots() -> dict[str, Path]:
    return {"results": Path("results"), "logs": Path("logs")}


def _iter_root_files(*, name: str, root: Path) -> Iterable[Path]:
    """Iterate files under a root while pruning irrelevant heavy subtrees."""
    if not root.exists():
        return

    root_resolved = root.resolve()
    for current, dirs, files in os.walk(root, topdown=True):
        current_path = Path(current)
        try:
            relative = current_path.resolve().relative_to(root_resolved)
        except ValueError:
            relative = Path()

        if name == "results":
            # Case-specific folders are already organized outputs and should not
            # be part of default artifact relocation scans.
            if relative.parts and relative.parts[0].startswith("case"):
                dirs[:] = []
                continue
            dirs[:] = [d for d in dirs if not d.startswith("case")]

        for filename in files:
            yield current_path / filename


def snapshot_default_artifacts(*, roots: dict[str, Path] | None = None) -> dict[str, set[Path]]:
    """Capture existing files in default results/logs roots."""
    roots = roots or _default_roots()
    snapshot: dict[str, set[Path]] = {}
    for name, root in roots.items():
        snapshot[name] = {path.resolve() for path in _iter_root_files(name=name, root=root)}
    return snapshot


def relocate_new_default_artifacts(
    *,
    snapshot: dict[str, set[Path]],
    output_dir: str,
    roots: dict[str, Path] | None = None,
) -> list[tuple[Path, Path]]:
    """Move new files generated in default roots into output_dir."""
    output_root = Path(output_dir).resolve()
    # Keep a consistent output layout across all case scripts.
    (output_root / "results").mkdir(parents=True, exist_ok=True)
    (output_root / "logs").mkdir(parents=True, exist_ok=True)
    roots = roots or _default_roots()
    moved: list[tuple[Path, Path]] = []

    for name, root in roots.items():
        if not root.exists():
            continue
        root_resolved = root.resolve()
        for path in _iter_root_files(name=name, root=root):
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
            moved_successfully = False
            try:
                shutil.move(str(source), str(target))
                moved_successfully = True
            except PermissionError:
                # On some platforms, moving an open file (e.g., active log) can fail.
                # Fallback to copy so artifacts are still captured in the case folder.
                try:
                    shutil.copy2(str(source), str(target))
                    moved_successfully = True
                except OSError:
                    moved_successfully = False
            except OSError:
                moved_successfully = False

            if moved_successfully:
                moved.append((source, target))

    return moved
