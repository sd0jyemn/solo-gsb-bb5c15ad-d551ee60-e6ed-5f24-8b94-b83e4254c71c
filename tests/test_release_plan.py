"""End-to-end tests for the read-only buffered-release planning endpoint."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("SENSOR_DB_PATH", db_path)
    from app import database, main

    main.app.state.db_path = db_path
    database.init_db(db_path)
    with TestClient(main.app) as c:
        yield c


def create_session(client, sources=("a", "b"), lateness=10.0, session_id="s"):
    r = client.post(
        "/sessions",
        json={
            "session_id": session_id,
            "source_ids": list(sources),
            "allowed_lateness_seconds": lateness,
        },
    )
    assert r.status_code == 201, r.text
    return session_id


def report(client, sid, readings):
    return client.post(f"/sessions/{sid}/readings", json={"readings": readings})


def seal(client, sid, source, final_observed_at):
    return client.post(
        f"/sessions/{sid}/sources/{source}/seal",
        json={"final_observed_at": final_observed_at},
    )


def plan(client, sid, count):
    return client.get(f"/sessions/{sid}/release-plan", params={"count": count})


def by_source(body):
    return {s["source_id"]: s for s in body["sources"]}


class TestReleasePlanBasics:
    def test_minimal_advancement_per_source(self, client):
        sid = create_session(client, lateness=10.0)
        report(client, sid, [
            {"source_id": "a", "seq": 1, "observed_at": "2026-01-01T00:00:00Z"},
            {"source_id": "a", "seq": 2, "observed_at": "2026-01-01T00:00:10Z"},
            {"source_id": "b", "seq": 1, "observed_at": "2026-01-01T00:00:20Z"},
        ])
        # Buffered global order: a@00, a@10, b@20.  Watermark is 00:00:00
        # (slowest a@10 minus 10s) so everything is still buffered.

        r = plan(client, sid, 1)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["session_id"] == sid
        assert body["count"] == 1
        assert body["buffered"] == 3
        assert body["allowed_lateness_seconds"] == 10.0
        # Target = 1st buffered observed_at + 1 microsecond.
        assert body["target_watermark"] == "2026-01-01T00:00:00.000001Z"
        assert body["attainable"] is True
        assert body["reason"] is None
        src = by_source(body)
        # required = target + 10s lateness.
        assert src["a"]["required"] == "2026-01-01T00:00:10.000001Z"
        assert src["b"]["required"] == "2026-01-01T00:00:10.000001Z"
        assert src["a"]["current"] == "2026-01-01T00:00:10Z"
        assert src["b"]["current"] == "2026-01-01T00:00:20Z"
        assert src["a"]["advance_seconds"] == pytest.approx(0.000001)
        assert src["b"]["advance_seconds"] == 0.0
        assert src["a"]["blocker"] is None
        assert src["b"]["blocker"] is None

    def test_nth_reading_defines_target(self, client):
        sid = create_session(client, lateness=10.0)
        report(client, sid, [
            {"source_id": "a", "seq": 1, "observed_at": "2026-01-01T00:00:00Z"},
            {"source_id": "a", "seq": 2, "observed_at": "2026-01-01T00:00:10Z"},
            {"source_id": "b", "seq": 1, "observed_at": "2026-01-01T00:00:20Z"},
        ])
        body = plan(client, sid, 3).json()
        assert body["target_watermark"] == "2026-01-01T00:00:20.000001Z"
        src = by_source(body)
        assert src["a"]["required"] == "2026-01-01T00:00:30.000001Z"
        assert src["a"]["advance_seconds"] == pytest.approx(20.000001)
        assert src["b"]["advance_seconds"] == pytest.approx(10.000001)

    def test_microsecond_precision_preserved(self, client):
        sid = create_session(client, lateness=2.5)
        report(client, sid, [
            {"source_id": "a", "seq": 1, "observed_at": "2026-01-01T00:00:05.123456Z"},
            {"source_id": "b", "seq": 1, "observed_at": "2026-01-01T00:00:09Z"},
        ])
        body = plan(client, sid, 1).json()
        assert body["target_watermark"] == "2026-01-01T00:00:05.123457Z"
        src = by_source(body)
        assert src["a"]["required"] == "2026-01-01T00:00:07.623457Z"
        assert src["a"]["advance_seconds"] == pytest.approx(2.500001)
        assert src["b"]["advance_seconds"] == 0.0

    def test_unreported_source_has_null_current(self, client):
        sid = create_session(client, lateness=10.0)
        report(client, sid, [
            {"source_id": "a", "seq": 1, "observed_at": "2026-01-01T00:00:05Z"},
        ])
        body = plan(client, sid, 1).json()
        assert body["attainable"] is True
        src = by_source(body)
        assert src["b"]["current"] is None
        assert src["b"]["advance_seconds"] is None
        assert src["b"]["blocker"] == "not_reported"
        assert src["b"]["required"] == "2026-01-01T00:00:15.000001Z"
        # The reported source still gets a concrete delta.
        assert src["a"]["advance_seconds"] == pytest.approx(10.000001)

    def test_plan_is_read_only_and_repeatable(self, client):
        sid = create_session(client, lateness=10.0)
        report(client, sid, [
            {"source_id": "a", "seq": 1, "observed_at": "2026-01-01T00:00:00Z"},
            {"source_id": "b", "seq": 1, "observed_at": "2026-01-01T00:00:20Z"},
        ])
        before = client.get(f"/sessions/{sid}").json()
        first = plan(client, sid, 2).json()
        second = plan(client, sid, 2).json()
        assert first == second
        after = client.get(f"/sessions/{sid}").json()
        assert before == after
        # Nothing was published by asking for a plan.
        assert client.get(f"/sessions/{sid}/stream").json()["readings"] == []


class TestReleasePlanSealedSources:
    def test_sealed_source_below_required_is_unattainable(self, client):
        sid = create_session(client, lateness=10.0)
        report(client, sid, [
            {"source_id": "a", "seq": 1, "observed_at": "2026-01-01T00:00:10Z"},
            {"source_id": "b", "seq": 1, "observed_at": "2026-01-01T00:00:20Z"},
        ])
        r = seal(client, sid, "b", "2026-01-01T00:00:20Z")
        assert r.status_code == 200, r.text

        body = plan(client, sid, 2).json()
        assert body["attainable"] is False
        assert "'b'" in body["reason"]
        assert "sealed" in body["reason"]
        src = by_source(body)
        # No advancement is suggested for a sealed source that falls short.
        assert src["b"]["current"] == "2026-01-01T00:00:20Z"
        assert src["b"]["required"] == "2026-01-01T00:00:30.000001Z"
        assert src["b"]["advance_seconds"] is None
        assert src["b"]["blocker"] == "sealed_below_required"
        # Open sources are still planned normally.
        assert src["a"]["advance_seconds"] == pytest.approx(20.000001)
        assert src["a"]["blocker"] is None

    def test_sealed_source_sufficient_stays_attainable(self, client):
        sid = create_session(client, lateness=10.0)
        report(client, sid, [
            {"source_id": "a", "seq": 1, "observed_at": "2026-01-01T00:00:10Z"},
            {"source_id": "b", "seq": 1, "observed_at": "2026-01-01T00:00:20Z"},
        ])
        seal(client, sid, "b", "2026-01-01T00:00:35Z")

        body = plan(client, sid, 2).json()
        assert body["attainable"] is True
        assert body["reason"] is None
        src = by_source(body)
        # The sealed final time is the source's effective clock.
        assert src["b"]["current"] == "2026-01-01T00:00:35Z"
        assert src["b"]["advance_seconds"] == 0.0
        assert src["b"]["blocker"] is None

    def test_sealed_shortfall_only_blocks_when_it_matters(self, client):
        sid = create_session(client, lateness=10.0)
        report(client, sid, [
            {"source_id": "a", "seq": 1, "observed_at": "2026-01-01T00:00:10Z"},
            {"source_id": "b", "seq": 1, "observed_at": "2026-01-01T00:00:20Z"},
        ])
        seal(client, sid, "b", "2026-01-01T00:00:20Z")

        # Releasing only the first reading needs required = 00:00:20.000001,
        # still above b's sealed 00:00:20 -> unattainable even for count=1.
        body = plan(client, sid, 1).json()
        assert body["attainable"] is False
        assert by_source(body)["b"]["blocker"] == "sealed_below_required"


class TestReleasePlanValidation:
    def test_count_bounds(self, client):
        sid = create_session(client)
        report(client, sid, [
            {"source_id": "a", "seq": 1, "observed_at": "2026-01-01T00:00:00Z"},
        ])
        assert plan(client, sid, 0).status_code == 422
        assert plan(client, sid, 101).status_code == 422
        assert client.get(f"/sessions/{sid}/release-plan").status_code == 422
        r = client.get(f"/sessions/{sid}/release-plan", params={"count": "x"})
        assert r.status_code == 422

    def test_count_beyond_buffered_is_422(self, client):
        sid = create_session(client)
        report(client, sid, [
            {"source_id": "a", "seq": 1, "observed_at": "2026-01-01T00:00:00Z"},
            {"source_id": "b", "seq": 1, "observed_at": "2026-01-01T00:00:20Z"},
        ])
        assert plan(client, sid, 2).status_code == 200
        r = plan(client, sid, 3)
        assert r.status_code == 422
        assert "buffered" in r.json()["detail"]

    def test_empty_backlog_is_422(self, client):
        sid = create_session(client)
        assert plan(client, sid, 1).status_code == 422

    def test_unknown_session_is_404(self, client):
        assert plan(client, "nope", 1).status_code == 404

    def test_closed_session_is_409(self, client):
        sid = create_session(client)
        report(client, sid, [
            {"source_id": "a", "seq": 1, "observed_at": "2026-01-01T00:00:10Z"},
            {"source_id": "b", "seq": 1, "observed_at": "2026-01-01T00:00:20Z"},
        ])
        seal(client, sid, "a", "2026-01-01T00:00:10Z")
        r = seal(client, sid, "b", "2026-01-01T00:00:20Z")
        assert r.json()["closed"] is True
        assert plan(client, sid, 1).status_code == 409


class TestReleasePlanExclusions:
    def test_published_and_quarantined_are_excluded(self, client):
        sid = create_session(client, lateness=10.0)
        report(client, sid, [
            {"source_id": "a", "seq": 1, "observed_at": "2026-01-01T00:00:00Z"},
            {"source_id": "b", "seq": 1, "observed_at": "2026-01-01T00:00:30Z"},
        ])
        # Advance the watermark to 00:00:20: a@00 publishes.
        r = report(client, sid, [
            {"source_id": "a", "seq": 2, "observed_at": "2026-01-01T00:00:40Z"},
        ]).json()
        assert r["published_total"] == 1
        # A late reading behind the published watermark is quarantined.
        r = report(client, sid, [
            {"source_id": "b", "seq": 0, "observed_at": "2026-01-01T00:00:10Z"},
        ]).json()
        assert r["quarantined"] == 1

        # Only the two buffered readings (b@30, a@40) are plannable.
        body = plan(client, sid, 2).json()
        assert body["buffered"] == 2
        assert body["target_watermark"] == "2026-01-01T00:00:40.000001Z"
        # The quarantined reading does not count towards the backlog.
        assert plan(client, sid, 3).status_code == 422
