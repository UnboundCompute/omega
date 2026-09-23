"""The seam is enforced by a test, not by discipline. M0_SPEC.md case 24.

DL-018 says exactly one module may import the private Rust extension. In the
reference system a facade with five bypassing callers turned one refactor into
1,148 compatibility re-exports across 332 files. The rule only holds if
breaking it is a build failure.

This check is AST-based, so it is not fooled by an alias, a ``from``-form, or a
dynamic ``import_module``. It is deliberately written so that the name it hunts
for never appears as a literal in this file — otherwise the checker would flag
itself.

**An import statement is not the only way in**, and matching only import
statements left two doors open. The extension becomes a live *attribute* of the
``omega`` package the moment anything imports ``omega.memory``, so ``import
omega`` followed by an attribute access needs no import of it at all; and a
module name that is computed rather than written leaves no constant to compare.
Both gave full frame-level access — raw append, offsets — with this suite
green. So the checker also reads attribute access and folds the strings it can,
and treats a module name the source does not fix as a violation rather than as
safe: a check that only sees the spellings someone thought of is not a check.
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


#: The two builtins that import a module named at runtime. What they import is
#: whatever their first argument says, so that argument is what gets read.
DYNAMIC_IMPORTERS = ("import_module", "__import__")


def _constant_str(node: ast.AST) -> str | None:
    """The string ``node`` evaluates to, when the source alone decides it.

    A plain constant folds, a ``+`` chain of constants folds, and an f-string
    with no substitutions folds — so ``"omega" + "." + "_log"`` is read for what
    it is rather than passed over for having no matching literal.

    Anything else returns ``None``, which every caller must read as
    **undecidable**, never as safe. That is the fail-closed half of the rule:
    the private module is exactly the one a bypass would compute.
    """
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _constant_str(node.left)
        right = _constant_str(node.right)
        return None if left is None or right is None else left + right
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            text = _constant_str(value)
            if text is None:
                return None
            parts.append(text)
        return "".join(parts)
    return None


def _call_reasons(node: ast.Call) -> list[str]:
    """Reaching the extension through a call rather than an import statement."""
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
    reasons: list[str] = []

    if name in DYNAMIC_IMPORTERS and node.args:
        if _constant_str(node.args[0]) is None:
            reasons.append(
                f"line {node.lineno}: {name}() with a module name the source does "
                f"not fix — it cannot be shown not to be {PRIVATE_MODULE}"
            )

    if name == "getattr" and len(node.args) >= 2:
        target = _constant_str(node.args[1])
        if target in (PRIVATE_LEAF, PRIVATE_MODULE):
            reasons.append(f"line {node.lineno}: getattr(..., {target!r})")

    return reasons


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

        elif isinstance(node, ast.Attribute):
            # The extension is a live attribute of the package as soon as
            # anything has imported the seam, so `import omega` and then
            # `omega._log.Log(p)` reaches it with no import of it anywhere.
            # Matched by name alone, exactly as the diagnostics hatch below is:
            # the seam is the one file that legitimately holds the handle, and
            # it is the one file excluded from this scan.
            if node.attr == PRIVATE_LEAF:
                reasons.append(f"line {node.lineno}: attribute access .{node.attr}")

        elif isinstance(node, ast.Call):
            reasons.extend(_call_reasons(node))

        else:
            # Catches importlib.import_module("omega._log"), __import__, and
            # the same name assembled from pieces. Exact match only: a
            # docstring that *mentions* the module is fine.
            text = _constant_str(node)
            if text is not None and text in (PRIVATE_MODULE, f".{PRIVATE_LEAF}"):
                reasons.append(f"line {node.lineno}: dynamic reference to {text!r}")

    return reasons


#: The seam's escape hatch for frame-level facts. Cases 34 and 35 assert things
#: about the offset index, which is vocabulary the seam exists to keep out of
#: the rest of the tree — so the hatch has to exist for the suite, and has to be
#: shut for everything else. A docstring saying "tests only" is the same kind of
#: convention that the rest of this file exists to replace, so it is a test too.
DIAGNOSTICS = "diagnostics"

#: The fsync counters are the same kind of thing: spec case 44 needs the
#: durability *call* observed, because `kill -9` cannot observe it, and that is
#: a fact about the write path rather than about episodes. They are imported
#: from the seam rather than reached through a store, so the detector has to
#: see the import form too.
HATCH_NAMES = (DIAGNOSTICS, "file_syncs", "dir_syncs")

#: Production source. Tests are deliberately excluded: they are the one caller
#: the hatch is for.
PRODUCTION_ROOT = "python"

#: The module the hatch is exported from — an import of a hatch name *from
#: here* is the form a plain attribute check would miss.
SEAM_MODULE = "omega.memory"


def _touches_diagnostics(path: Path) -> list[str]:
    """Every way ``path`` reaches the diagnostics hatch, as readable reasons."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    reasons: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in HATCH_NAMES:
            reasons.append(f"line {node.lineno}: attribute access .{node.attr}")
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in HATCH_NAMES
        ):
            reasons.append(f"line {node.lineno}: getattr(..., {node.args[1].value!r})")
        elif isinstance(node, ast.ImportFrom) and (node.module or "") == SEAM_MODULE:
            for alias in node.names:
                if alias.name in HATCH_NAMES:
                    reasons.append(
                        f"line {node.lineno}: from {SEAM_MODULE} import {alias.name}"
                    )

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
        "fsync-import": "from omega.memory import file_syncs\n",
        "dirsync-import": "from omega.memory import MemoryStore, dir_syncs\n",
        "fsync-attribute": "import omega.memory\nomega.memory.file_syncs()\n",
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


def test_case_24_the_checker_catches_the_bypasses_an_import_scan_misses(
    tmp_path: Path,
) -> None:
    """Case 24 — the two doors that stayed open while this suite was green.

    Both give exactly what the seam exists to withhold: the raw handle, and
    with it frame-level append and the offset index. Neither needs an import
    *statement* naming the extension, which is all the checker used to read.

    1. The extension is an attribute of the ``omega`` package as soon as
       anything imports ``omega.memory`` — which importing the seam does. So
       ``import omega`` and then ``omega._log.Log(p)`` is a complete bypass.
    2. A computed module name has no constant to compare against, so
       ``import_module("omega" + "." + "_log")`` read as an ordinary call.

    Asserted against the checker on source snippets rather than against the
    tree, because the tree is clean either way: "the real files do not do this"
    passes just as well when the detector is blind, which is how these two
    survived. This is the detector's control, and it is the test that matters.
    """
    p = "path"
    bypasses = {
        # 1 - attribute access on the package, in its plain, aliased and
        #     through-the-seam spellings.
        "package-attribute": f"import {PACKAGE}\nraw = {PACKAGE}.{PRIVATE_LEAF}.Log({p})\n",
        "package-attribute-aliased": (
            f"import {PACKAGE} as pkg\nraw = pkg.{PRIVATE_LEAF}.Log({p})\n"
        ),
        "submodule-attribute": (
            f"import {PACKAGE}.memory\nraw = {PACKAGE}.memory.{PRIVATE_LEAF}.Log({p})\n"
        ),
        "through-a-store": (
            f"from {PACKAGE}.memory import MemoryStore\n"
            f"raw = MemoryStore.open({p}).{PRIVATE_LEAF}\n"
        ),
        "getattr-on-the-package": (
            f"import {PACKAGE}\nraw = getattr({PACKAGE}, {PRIVATE_LEAF!r})\n"
        ),
        # 2 - the module name assembled, or simply not written down.
        "concatenated-name": (
            "import importlib\n"
            f"raw = importlib.import_module({PACKAGE!r} + '.' + {PRIVATE_LEAF!r})\n"
        ),
        "name-from-a-variable": (
            "import importlib\n"
            f"leaf = {PRIVATE_LEAF!r}\n"
            f"raw = importlib.import_module({PACKAGE!r} + '.' + leaf)\n"
        ),
        "dunder-import-concatenated": (
            f"raw = __import__({PACKAGE!r} + '.' + {PRIVATE_LEAF!r})\n"
        ),
        "relative-dynamic-import": (
            "import importlib\n"
            f"raw = importlib.import_module('.' + {PRIVATE_LEAF!r}, {PACKAGE!r})\n"
        ),
    }
    for name, source in bypasses.items():
        probe = tmp_path / f"bypass_{name}.py"
        probe.write_text(source, encoding="utf-8")
        assert _imports_private(probe), f"{name} bypass was not detected"

    # And the widened net still does not catch ordinary code. A similar name is
    # not the name, and an import of the seam is what every caller should do.
    allowed = {
        "similar-attribute": "self._logger.info(x)\nself._log_path = p\n",
        "similar-constant": "name = 'omega.logging'\nleaf = 'log'\n",
        "seam-usage": (
            "from omega.memory import MemoryStore\n"
            "with MemoryStore.open(p) as s:\n    s.append_episode(b'x')\n"
        ),
        "unrelated-dynamic-import": (
            "import importlib\nmod = importlib.import_module('json')\n"
        ),
    }
    for name, source in allowed.items():
        probe = tmp_path / f"okbypass_{name}.py"
        probe.write_text(source, encoding="utf-8")
        assert _imports_private(probe) == [], f"{name} was a false positive"
