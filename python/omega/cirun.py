"""A CI thread's latest run, read from GitHub (DL-121 A1). Read-only.

A CI key opens on a failure mail and used to stay open until a thinking slot
let it go, so a workflow that went green again was still told as failing.
Before composing, the executor asks here for the workflow's newest completed
run; a green one newer than the failure closes the key.

Two GETs at most, against GitHub's REST API, with the token the GitHub MCP
server is already given (``OMEGA_MCP_GITHUB_TOKEN``) when there is one; a
public repo answers without. **Fails closed**: anything that cannot be read
(no owner in the line, a network error, a refusal, an answer of the wrong
shape) is ``None``, "couldn't check", and the key stays open.
"""

from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

__all__ = ["Run", "latest", "API", "TOKEN_ENV"]

API = "https://api.github.com"
TOKEN_ENV = "OMEGA_MCP_GITHUB_TOKEN"
TIMEOUT_SECONDS = 10
DEFAULT_BRANCHES = ("main", "master")

_WHAT = re.compile(r"^CI failed: (?P<wf>.+) on (?P<owner>[A-Za-z0-9_.-]{1,100})/(?P<repo>[A-Za-z0-9_.-]{1,100})$")
_RUN = re.compile(r"^https://github\.com/(?P<owner>[A-Za-z0-9_.-]{1,100})/(?P<repo>[A-Za-z0-9_.-]{1,100})/actions/runs/(?P<id>[0-9]{1,20})$")


@dataclass(frozen=True)
class Run:
    """A workflow's newest completed run."""

    conclusion: str
    at: datetime
    url: Optional[str] = None


def _get(path: str, env: Mapping[str, str]) -> Any:
    req = urllib.request.Request(f"{API}{path}", headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "omega",
        "X-GitHub-Api-Version": "2022-11-28",
    })
    token = env.get(TOKEN_ENV, "").strip()
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:  # noqa: S310 - fixed https host
        return json.loads(resp.read().decode("utf-8"))


def _run(raw: Any) -> Optional[Run]:
    if not isinstance(raw, dict) or not isinstance(raw.get("conclusion"), str):
        return None
    try:
        at = datetime.fromisoformat(str(raw.get("updated_at") or raw.get("created_at")).replace("Z", "+00:00"))
    except ValueError:
        return None
    url = raw.get("html_url") if isinstance(raw.get("html_url"), str) else None
    return Run(conclusion=raw["conclusion"], at=at if at.tzinfo else at.replace(tzinfo=timezone.utc), url=url)


def latest(item: Any, env: Mapping[str, str] = os.environ) -> Optional[Run]:
    """The newest completed run of the workflow ``item`` (a ``ci:`` ledger
    item) failed on, on the failing run's branch; ``None`` when it cannot be
    determined. Never raises."""
    try:
        m = _WHAT.match(str(getattr(item, "what", "")))
        if m is None:
            return None
        owner, repo, workflow = m.group("owner"), m.group("repo"), m.group("wf").strip()
        q = urllib.parse.quote
        base = f"/repos/{q(owner)}/{q(repo)}/actions"
        link = _RUN.match(str(getattr(item, "url", "") or ""))
        if link is not None and (link.group("owner"), link.group("repo")) == (owner, repo):
            failed = _get(f"{base}/runs/{link.group('id')}", env)
            wid, branch = failed.get("workflow_id"), failed.get("head_branch")
            if not isinstance(wid, int) or not isinstance(branch, str):
                return None
            found = _get(f"{base}/workflows/{wid}/runs?status=completed&per_page=1"
                         f"&branch={q(branch)}", env)
            runs = found.get("workflow_runs") if isinstance(found, dict) else None
            return _run(runs[0]) if isinstance(runs, list) and runs else None
        found = _get(f"{base}/runs?status=completed&per_page=30", env)
        runs = found.get("workflow_runs") if isinstance(found, dict) else None
        if not isinstance(runs, list):
            return None
        for raw in runs:  # newest first; with no failing run to name a
            # branch, only the default branch's runs speak for the workflow
            if (isinstance(raw, dict)
                    and str(raw.get("name", "")).casefold() == workflow.casefold()
                    and raw.get("head_branch") in DEFAULT_BRANCHES):
                return _run(raw)
        return None
    except Exception:  # noqa: BLE001 - fails closed: "couldn't check"
        return None
