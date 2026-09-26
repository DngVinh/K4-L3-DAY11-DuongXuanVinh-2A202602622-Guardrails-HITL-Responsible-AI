"""
Assignment 11 — Audit log implementation.

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, dict] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Store an input and its start time until the output is available."""
        key = request_id or user_id
        self._open[key] = {
            "request_id": request_id or key,
            "user_id": user_id,
            "input": text,
            "input_timestamp": utc_now_iso(),
            "started_at": time.perf_counter(),
        }

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Complete a pending interaction and append it to the audit log."""
        key = request_id or user_id
        pending = self._open.pop(key, None)
        finished_at = time.perf_counter()
        started_at = pending.get("started_at", finished_at) if pending else finished_at

        self.logs.append({
            "request_id": (
                pending.get("request_id") if pending else (request_id or key)
            ),
            "user_id": pending.get("user_id", user_id) if pending else user_id,
            "input": pending.get("input", "") if pending else "",
            "input_timestamp": (
                pending.get("input_timestamp") if pending else utc_now_iso()
            ),
            "output_timestamp": utc_now_iso(),
            "output": text,
            "blocked": bool(blocked),
            "layer": layer,
            "latency_ms": round((finished_at - started_at) * 1000, 3),
        })

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path()).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
