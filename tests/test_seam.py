"""The seam is enforced by a test, not by discipline. M0_SPEC.md case 24.

DL-018 says exactly one module may import the private Rust extension. In the
reference system a facade with five bypassing callers turned one refactor into
1,148 compatibility re-exports across 332 files. The rule only holds if
breaking it is a build failure.

This check is AST-based, so it is not fooled by an alias, a ``from``-form, or a
dynamic ``import_module``. It is deliberately written so that the name it hunts
for never appears as a literal in this file — otherwise the checker would flag
itself.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

PACKAGE = "omega"
PRIVATE_LEAF = "_log"
PRIVATE_MODULE = f"{PACKAGE}.{PRIVATE_LEAF}"

#: The one file allowed to import it, relative to the repo root.
THE_SEAM = Path("python/omega/memory/__init__.py")

#: Where source that must obey the rule lives.
SCAN_ROOTS = ("python", "tests")

SKIP_DIRS = {"__pycache__", ".venv", "target", "build", "dist", ".git"}


def _python_files() -> list[Path]:
    found: list[Path] = []
    for root in SCAN_ROOTS:
        base = REPO_ROOT / root
        assert base.is_dir(), f"scan root {base} does not exist"
        for path in sorted(base.rglob("*.py")):
            if SKIP_DIRS & set(path.parts):
                continue
            found.append(path)
    return found


def _imports_private(path: Path) -> list[str]:
    """Every way ``path`` reaches the private extension, as readable reasons."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    reasons: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == PRIVATE_MODULE or alias.name.startswith(
                    PRIVATE_MODULE + "."
                ):
                    reasons.append(f"line {node.lineno}: import {alias.name}")

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = [a.name for a in node.names]
            if module == PRIVATE_MODULE or module.startswith(PRIVATE_MODULE + "."):
                reasons.append(f"line {node.lineno}: from {module} import ...")
            elif module == PACKAGE and PRIVATE_LEAF in names:
                reasons.append(f"line {node.lineno}: from {PACKAGE} import {PRIVATE_LEAF}")
            elif node.level > 0 and PRIVATE_LEAF in names:
                reasons.append(
                    f"line {node.lineno}: relative import of {PRIVATE_LEAF}"
                )
            elif node.level > 0 and module.split(".")[-1] == PRIVATE_LEAF:
                reasons.append(f"line {node.lineno}: relative from .{module} import ...")

        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # Catches importlib.import_module("omega._log") and __import__ too.
            # Exact match only: a docstring that *mentions* the module is fine.
            if node.value == PRIVATE_MODULE:
                reasons.append(
                    f"line {node.lineno}: dynamic reference to {PRIVATE_MODULE!r}"
                )

    return reasons


#: The seam's escape hatch for frame-level facts. Cases 34 and 35 assert things
#: about the offset index, which is vocabulary the seam exists to keep out of
#: the rest of the tree — so the hatch has to exist for the suite, and has to be
#: shut for everything else. A docstring saying "tests only" is the same kind of
#: convention that the rest of this file exists to replace, so it is a test too.
DIAGNOSTICS = "diagnostics"

#: Production source. Tests are deliberately excluded: they are the one caller
#: the hatch is for.
PRODUCTION_ROOT = "python"


def _touches_diagnostics(path: Path) -> list[str]:
    """Every way ``path`` reaches the diagnostics hatch, as readable reasons."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    reasons: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == DIAGNOSTICS:
            reasons.append(f"line {node.lineno}: attribute access .{DIAGNOSTICS}")
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == DIAGNOSTICS
        ):
            reasons.append(f"line {node.lineno}: getattr(..., {DIAGNOSTICS!r})")

    return reasons


def test_case_24_production_code_does_not_use_the_diagnostics_hatch() -> None:
    """Case 24 — the hatch is open to the suite and shut to everything else.

    ``diagnostics`` exposes offsets and index rebuilds: frame vocabulary that
    the seam is otherwise built to stop. It is the one hole in the seam, so it
    is the one most worth a test. The seam itself is skipped because it defines
    the hatch.
    """
    base = REPO_ROOT / PRODUCTION_ROOT
    assert base.is_dir(), f"production root {base} does not exist"

    files = [
        p
        for p in sorted(base.rglob("*.py"))
        if not SKIP_DIRS & set(p.parts) and p.relative_to(REPO_ROOT) != THE_SEAM
    ]

    violations: dict[str, list[str]] = {}
    for path in files:
        reasons = _touches_diagnostics(path)
        if reasons:
            violations[str(path.relative_to(REPO_ROOT))] = reasons

    assert not violations, (
        f"MemoryStore.{DIAGNOSTICS} is for the M0 suite only — it speaks frames, "
        f"not episodes. Offenders:\n"
        + "\n".join(f"  {f}: {'; '.join(r)}" for f, r in sorted(violations.items()))
    )


def test_case_24_the_diagnostics_detector_is_not_a_no_op(tmp_path: Path) -> None:
    """Case 24 — its control. An empty walk over production code is the expected
    result, so the detector must be shown to fire on something."""
    caught = {
        "attribute": "store.diagnostics.offsets()\n",
        "chained": "get_store().diagnostics.rebuild_indexes()\n",
        "getattr": "getattr(store, 'diagnostics')\n",
    }
    for name, source in caught.items():
        probe = tmp_path / f"diag_{name}.py"
        probe.write_text(source, encoding="utf-8")
        assert _touches_diagnostics(probe), f"{name} form was not detected"

    allowed = {
        "episode-level": "store.append_episode(b'x')\nstore.head()\n",
        "similar-name": "store.diagnostics_report\nx = 'diagnostics'\n",
    }
    for name, source in allowed.items():
        probe = tmp_path / f"okdiag_{name}.py"
        probe.write_text(source, encoding="utf-8")
        assert _touches_diagnostics(probe) == [], f"{name} was a false positive"


def test_case_24_only_the_seam_imports_the_private_extension() -> None:
    """Case 24 — any module outside omega.memory importing the extension fails."""
    files = _python_files()
    # Fail closed on empty: a walker that found nothing would pass vacuously.
    assert len(files) >= 5, f"only found {len(files)} python files to scan: {files}"

    violations: dict[str, list[str]] = {}
    for path in files:
        rel = path.relative_to(REPO_ROOT)
        if rel == THE_SEAM:
            continue
        reasons = _imports_private(path)
        if reasons:
            violations[str(rel)] = reasons

    assert not violations, (
        f"{PRIVATE_MODULE} may only be imported by {THE_SEAM}. Offenders:\n"
        + "\n".join(f"  {f}: {'; '.join(r)}" for f, r in sorted(violations.items()))
    )


def test_case_24_the_checker_actually_sees_the_seams_own_import() -> None:
    """Case 24 — and the detector is not a no-op: it finds the one legal import.

    If this went green while the detector was broken, the check above would be
    a check that passes on empty. This is its control.
    """
    seam = REPO_ROOT / THE_SEAM
    assert seam.is_file(), f"the seam is missing at {seam}"
    reasons = _imports_private(seam)
    assert reasons, f"the seam does not appear to import {PRIVATE_MODULE} at all"


def test_case_24_the_checker_catches_every_import_form(tmp_path: Path) -> None:
    """Case 24 — each way of reaching the extension is detected, and normal
    imports of the seam are not false positives."""
    caught = {
        "plain": f"import {PRIVATE_MODULE}\n",
        "aliased": f"import {PRIVATE_MODULE} as raw\n",
        "from-module": f"from {PRIVATE_MODULE} import Log\n",
        "from-package": f"from {PACKAGE} import {PRIVATE_LEAF}\n",
        "relative": f"from . import {PRIVATE_LEAF}\n",
        "dynamic": f"import importlib\nraw = importlib.import_module({PRIVATE_MODULE!r})\n",
    }
    for name, source in caught.items():
        probe = tmp_path / f"probe_{name}.py"
        probe.write_text(source, encoding="utf-8")
        assert _imports_private(probe), f"{name} form was not detected"

    allowed = {
        "seam-import": "from omega.memory import MemoryStore\n",
        "mention-in-docstring": f'"""talks about {PRIVATE_MODULE} but does not import it."""\n',
        "unrelated": "import os\nimport logging\n",
    }
    for name, source in allowed.items():
        probe = tmp_path / f"ok_{name}.py"
        probe.write_text(source, encoding="utf-8")
        assert _imports_private(probe) == [], f"{name} was a false positive"
