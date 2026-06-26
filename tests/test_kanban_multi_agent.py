"""
Tests for multi-agent Kanban features: request_review, claim_audit, poll_tasks,
auditor field, and the kanban_request_review / kanban_poll agent tools.
"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# DB-layer tests
# ---------------------------------------------------------------------------

@pytest.fixture
def fresh_db(monkeypatch, tmp_path):
    """Isolated Hermes home with a fresh kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-editor")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    from pathlib import Path as _Path
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return kb


def test_auditor_column_exists(fresh_db):
    """The auditor column should exist after init_db."""
    conn = fresh_db.connect()
    try:
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)")}
        assert "auditor" in cols
    finally:
        conn.close()


def test_task_has_auditor_field(fresh_db):
    """Task dataclass should expose auditor, defaulting to None."""
    conn = fresh_db.connect()
    try:
        tid = fresh_db.create_task(conn, title="test", assignee="editor")
        task = fresh_db.get_task(conn, tid)
        assert task.auditor is None
    finally:
        conn.close()


def test_request_review_transition(fresh_db):
    """request_review should move running -> review and set auditor."""
    conn = fresh_db.connect()
    try:
        tid = fresh_db.create_task(conn, title="work", assignee="editor")
        fresh_db.claim_task(conn, tid)
        assert fresh_db.get_task(conn, tid).status == "running"

        ok = fresh_db.request_review(
            conn, tid, auditor="auditor-profile", reason="needs check"
        )
        assert ok is True

        task = fresh_db.get_task(conn, tid)
        assert task.status == "review"
        assert task.auditor == "auditor-profile"
        assert task.claim_lock is None
    finally:
        conn.close()


def test_request_review_no_auditor(fresh_db):
    """request_review without auditor should set auditor to None."""
    conn = fresh_db.connect()
    try:
        tid = fresh_db.create_task(conn, title="work", assignee="editor")
        fresh_db.claim_task(conn, tid)
        ok = fresh_db.request_review(conn, tid)
        assert ok is True
        task = fresh_db.get_task(conn, tid)
        assert task.status == "review"
        assert task.auditor is None
    finally:
        conn.close()


def test_request_review_fails_on_non_running(fresh_db):
    """request_review should fail if task is not running."""
    conn = fresh_db.connect()
    try:
        tid = fresh_db.create_task(conn, title="work", assignee="editor")
        # Task is in 'ready' status, not 'running'
        ok = fresh_db.request_review(conn, tid)
        assert ok is False
    finally:
        conn.close()


def test_claim_audit_success(fresh_db):
    """claim_audit should atomically claim a review task."""
    conn = fresh_db.connect()
    try:
        tid = fresh_db.create_task(conn, title="work", assignee="editor")
        fresh_db.claim_task(conn, tid)
        fresh_db.request_review(conn, tid, auditor="auditor-1")

        claimed = fresh_db.claim_audit(conn, tid, claimer="host:123")
        assert claimed is not None
        assert claimed.id == tid
        assert claimed.claim_lock == "host:123"
        assert claimed.status == "review"  # status stays review
    finally:
        conn.close()


def test_claim_audit_fails_on_non_review(fresh_db):
    """claim_audit should fail if task is not in review."""
    conn = fresh_db.connect()
    try:
        tid = fresh_db.create_task(conn, title="work", assignee="editor")
        # Task is in 'ready', not 'review'
        claimed = fresh_db.claim_audit(conn, tid)
        assert claimed is None
    finally:
        conn.close()


def test_claim_audit_atomic(fresh_db):
    """Second claim_audit on same task should fail (CAS)."""
    conn = fresh_db.connect()
    try:
        tid = fresh_db.create_task(conn, title="work", assignee="editor")
        fresh_db.claim_task(conn, tid)
        fresh_db.request_review(conn, tid)

        first = fresh_db.claim_audit(conn, tid, claimer="host:111")
        assert first is not None

        second = fresh_db.claim_audit(conn, tid, claimer="host:222")
        assert second is None  # already claimed
    finally:
        conn.close()


def test_poll_editor(fresh_db):
    """poll_tasks(role=editor) should list ready unclaimed tasks."""
    conn = fresh_db.connect()
    try:
        t1 = fresh_db.create_task(conn, title="task1", assignee="ed", priority=5)
        t2 = fresh_db.create_task(conn, title="task2", assignee="ed", priority=1)
        # Claim t1 so only t2 is available
        fresh_db.claim_task(conn, t1)

        tasks = fresh_db.poll_tasks(conn, role="editor", limit=10)
        ids = [t["id"] for t in tasks]
        assert t2 in ids
        assert t1 not in ids  # claimed, not available
    finally:
        conn.close()


def test_poll_auditor(fresh_db):
    """poll_tasks(role=auditor) should list review unclaimed tasks."""
    conn = fresh_db.connect()
    try:
        t1 = fresh_db.create_task(conn, title="work1", assignee="ed")
        t2 = fresh_db.create_task(conn, title="work2", assignee="ed")
        fresh_db.claim_task(conn, t1)
        fresh_db.claim_task(conn, t2)
        fresh_db.request_review(conn, t1, auditor="aud-A")
        fresh_db.request_review(conn, t2)

        # No profile filter: all review tasks
        tasks = fresh_db.poll_tasks(conn, role="auditor", limit=10)
        ids = [t["id"] for t in tasks]
        assert set(ids) == {t1, t2}

        # Profile filter: only tasks assigned to aud-A or unassigned
        tasks_a = fresh_db.poll_tasks(conn, role="auditor", profile="aud-A", limit=10)
        ids_a = [t["id"] for t in tasks_a]
        assert t1 in ids_a  # auditor=aud-A
        assert t2 in ids_a  # auditor=None (unassigned)
    finally:
        conn.close()


def test_poll_auditor_profile_filter(fresh_db):
    """poll_tasks(role=auditor, profile=X) should exclude tasks assigned to other auditors."""
    conn = fresh_db.connect()
    try:
        t1 = fresh_db.create_task(conn, title="work1", assignee="ed")
        t2 = fresh_db.create_task(conn, title="work2", assignee="ed")
        fresh_db.claim_task(conn, t1)
        fresh_db.claim_task(conn, t2)
        fresh_db.request_review(conn, t1, auditor="aud-A")
        fresh_db.request_review(conn, t2, auditor="aud-B")

        # aud-A should only see t1, not t2
        tasks = fresh_db.poll_tasks(conn, role="auditor", profile="aud-A", limit=10)
        ids = [t["id"] for t in tasks]
        assert t1 in ids
        assert t2 not in ids
    finally:
        conn.close()


def test_poll_invalid_role(fresh_db):
    """poll_tasks should reject invalid role."""
    conn = fresh_db.connect()
    try:
        with pytest.raises(ValueError, match="role must be"):
            fresh_db.poll_tasks(conn, role="invalid")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Tool handler tests
# ---------------------------------------------------------------------------

@pytest.fixture
def worker_env(monkeypatch, tmp_path):
    """Worker session with a claimed task ready for review."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    from pathlib import Path as _Path
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="worker-test", assignee="test-worker")
        kb.claim_task(conn, tid)
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    return tid


def test_kanban_request_review_tool(worker_env):
    """kanban_request_review tool should transition to review."""
    from tools import kanban_tools as kt
    out = kt._handle_request_review({
        "auditor": "forensic-auditor",
        "reason": "needs independent verification",
    })
    d = json.loads(out)
    assert d.get("ok") is True
    assert d["status"] == "review"
    assert d["auditor"] == "forensic-auditor"


def test_kanban_request_review_no_auditor(worker_env):
    """kanban_request_review without auditor should work."""
    from tools import kanban_tools as kt
    out = kt._handle_request_review({})
    d = json.loads(out)
    assert d.get("ok") is True
    assert d["status"] == "review"


def test_kanban_request_review_fails_on_non_running(monkeypatch, tmp_path):
    """kanban_request_review should fail if task is not running."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    from pathlib import Path as _Path
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="test", assignee="test-worker")
    finally:
        conn.close()
    # Don't claim - task stays in 'ready'
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)

    from tools import kanban_tools as kt
    out = kt._handle_request_review({})
    d = json.loads(out)
    assert "error" in d


def test_kanban_poll_tool_editor(monkeypatch, tmp_path):
    """kanban_poll tool should list ready tasks for editor role."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "orchestrator")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    from pathlib import Path as _Path
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kb.connect()
    try:
        t1 = kb.create_task(conn, title="ready-task", assignee="ed")
    finally:
        conn.close()

    from tools import kanban_tools as kt
    out = kt._handle_poll({"role": "editor", "limit": 10})
    d = json.loads(out)
    assert d.get("ok") is True
    assert d["role"] == "editor"
    assert d["count"] >= 1
    ids = [t["id"] for t in d["tasks"]]
    assert t1 in ids


def test_kanban_poll_tool_auditor(monkeypatch, tmp_path):
    """kanban_poll tool should list review tasks for auditor role."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "orchestrator")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    from pathlib import Path as _Path
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="review-task", assignee="ed")
        kb.claim_task(conn, tid)
        kb.request_review(conn, tid, auditor="aud")
    finally:
        conn.close()

    from tools import kanban_tools as kt
    out = kt._handle_poll({"role": "auditor", "profile": "aud", "limit": 10})
    d = json.loads(out)
    assert d.get("ok") is True
    assert d["role"] == "auditor"
    assert d["count"] >= 1
    ids = [t["id"] for t in d["tasks"]]
    assert tid in ids


def test_kanban_poll_tool_invalid_role():
    """kanban_poll tool should reject invalid role."""
    from tools import kanban_tools as kt
    out = kt._handle_poll({"role": "invalid"})
    d = json.loads(out)
    assert "error" in d

def test_request_review_preserves_handoff_metadata(fresh_db):
    """request_review should preserve the editor handoff for the auditor."""
    conn = fresh_db.connect()
    try:
        tid = fresh_db.create_task(conn, title="work", assignee="editor")
        fresh_db.claim_task(conn, tid)
        ok = fresh_db.request_review(
            conn,
            tid,
            auditor="auditor-profile",
            reason="needs check",
            summary="editor handoff",
            metadata={"changed_files": ["a.py"]},
            verified_cards=["t_child1234"],
        )
        assert ok is True
        run = fresh_db.latest_run(conn, tid)
        assert run is not None
        assert run.outcome == "review_requested"
        assert run.summary == "editor handoff"
        assert run.metadata == {"changed_files": ["a.py"]}
        event = [e for e in fresh_db.list_events(conn, tid) if e.kind == "review_requested"][-1]
        assert event.payload["verified_cards"] == ["t_child1234"]
    finally:
        conn.close()


def test_complete_task_needs_audit_routes_to_review(fresh_db):
    """complete_task should auto-route worker handoffs to review when requested."""
    conn = fresh_db.connect()
    try:
        tid = fresh_db.create_task(conn, title="work", assignee="editor")
        fresh_db.claim_task(conn, tid)
        ok = fresh_db.complete_task(
            conn,
            tid,
            summary="editor handoff",
            metadata={
                "needs_audit": True,
                "audit_profile": "forensic-auditor",
                "audit_reason": "verify facts",
                "findings": ["one", "two"],
            },
        )
        assert ok is True
        task = fresh_db.get_task(conn, tid)
        assert task.status == "review"
        assert task.auditor == "forensic-auditor"
        run = fresh_db.latest_run(conn, tid)
        assert run is not None
        assert run.outcome == "review_requested"
        assert run.summary == "editor handoff"
        assert run.metadata["needs_audit"] is True
        assert run.metadata["findings"] == ["one", "two"]
    finally:
        conn.close()


def test_kanban_complete_needs_audit_auto_reviews(worker_env):
    """kanban_complete should surface review status when metadata requests audit."""
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    out = kt._handle_complete({
        "summary": "editor handoff",
        "metadata": {
            "needs_audit": True,
            "audit_profile": "forensic-auditor",
            "audit_reason": "verify facts",
        },
    })
    d = json.loads(out)
    assert d.get("ok") is True
    assert d["status"] == "review"
    assert d["auditor"] == "forensic-auditor"

    conn = kb.connect()
    try:
        task = kb.get_task(conn, worker_env)
        assert task.status == "review"
        assert task.auditor == "forensic-auditor"
    finally:
        conn.close()


def test_cli_complete_needs_audit_auto_reviews(monkeypatch, tmp_path, capsys):
    """CLI complete should print review handoff when metadata requests audit."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    from pathlib import Path as _Path
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)

    from hermes_cli import kanban as cli
    from hermes_cli import kanban_db as kb
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="cli-test", assignee="test-worker")
        kb.claim_task(conn, tid)
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    args = SimpleNamespace(
        task_ids=[tid],
        summary="cli handoff",
        metadata=json.dumps({
            "needs_audit": True,
            "audit_profile": "cli-auditor",
        }),
        result=None,
    )
    rc = cli._cmd_complete(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "Requested review for" in out
    conn = kb.connect()
    try:
        task = kb.get_task(conn, tid)
        assert task.status == "review"
        assert task.auditor == "cli-auditor"
    finally:
        conn.close()

