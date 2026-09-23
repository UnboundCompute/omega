"""Marks ``tests`` as a package so ``from tests.teaching import ...`` resolves.

Without this file the shared helpers imported only under ``python -m pytest``,
which prepends the working directory to ``sys.path``. The bare ``pytest``
console script — what CI runs — does not, so collection failed there while
every local run passed. Making this a package puts the *repo root* on the path
under pytest's prepend import mode instead of ``tests/``, which fixes both
invocations and gives ``tests.teaching`` a single module identity rather than
letting it be importable under two names.
"""
