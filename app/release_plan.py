"""Read-only release planning for the earliest buffered readings.

``GET /sessions/{session_id}/release-plan?count=N`` computes, inside one
SQLite read snapshot, the minimal per-source clock advancement that would
release the earliest ``N`` buffered readings of an *open* session in the
deterministic global order ``(observed_at, source_id, seq)``.  Published and
quarantined readings never take part.

The target watermark is one microsecond past the Nth buffered reading's
``observed_at`` -- the smallest watermark that publishes it, because
publication is strictly ``observed_at < watermark`` and stored timestamps
have microsecond resolution.  Adding the session's
``allowed_lateness_seconds`` yields the *required effective max*: the event
time every source's effective clock (its declared ``final_observed_at`` once
sealed, otherwise its largest reported ``observed_at``) must reach for the
slowest-source watermark to hit that target.

The plan is a pure computation and mutates nothing.  A source that has not
reported yet has ``current = null``.  A *sealed* source whose declared final
time is below the required effective max can never be advanced again, so
instead of suggesting an impossible advancement the plan reports
``attainable = false`` together with the reason.
"""
from __future__ import annotations

import sqlite3
from datetime import timedelta
from typing import Any

from .database import read_transaction
from .errors import ConflictError, UnprocessableEntityError
from .service import _load_session, _source_rows
from .time_utils import format_event_time, parse_event_time, store_event_time

__all__ = ["compute_release_plan"]

# Publication is strictly ``observed_at < watermark``; one microsecond past
# the Nth reading's event time is the smallest watermark that releases it.
_TARGET_EPSILON = timedelta(microseconds=1)


def compute_release_plan(
    conn: sqlite3.Connection, session_id: str, count: int
) -> dict[str, Any]:
    """Compute the minimal clock-advancement plan releasing ``count`` readings.

    Everything is read inside a single deferred transaction, so the buffered
    scan, the per-source clocks and the session row come from one consistent
    SQLite snapshot.  Raises 404 for an unknown session, 409 for a closed
    session and 422 when ``count`` exceeds the buffered backlog.
    """
    with read_transaction(conn):
        session = _load_session(conn, session_id)
        if session["status"] == "closed":
            raise ConflictError(
                f"session {session_id!r} is closed: no buffered readings remain"
            )

        lateness = session["allowed_lateness_seconds"]
        buffered = conn.execute(
            "SELECT COUNT(*) AS c FROM readings"
            " WHERE session_id = ? AND status = 'buffered'",
            (session_id,),
        ).fetchone()["c"]
        if count > buffered:
            raise UnprocessableEntityError(
                f"count {count} exceeds the number of buffered readings"
                f" ({buffered}) in session {session_id!r}"
            )

        # The Nth earliest buffered reading in the deterministic global order
        # defines the target watermark.
        nth = conn.execute(
            "SELECT observed_at FROM readings"
            " WHERE session_id = ? AND status = 'buffered'"
            " ORDER BY observed_at, source_id, seq LIMIT 1 OFFSET ?",
            (session_id, count - 1),
        ).fetchone()
        target = store_event_time(parse_event_time(nth["observed_at"]) + _TARGET_EPSILON)
        # WM = min(effective source max) - lateness >= target  <=>  every
        # source's effective max reaches target + lateness.
        required = store_event_time(
            parse_event_time(target) + timedelta(seconds=lateness)
        )

        sources: list[dict[str, Any]] = []
        shortfalls: list[str] = []
        for row in _source_rows(conn, session_id):
            sid = row["source_id"]
            # A sealed source's effective clock is its declared final time.
            sealed = row["final_observed_at"] is not None
            current = row["final_observed_at"] or row["max_observed_at"]

            blocker = None
            advance_seconds = None
            if current is None:
                # Never reported: its clock is undefined, so no delta can be
                # computed; the first report must land at/above ``required``.
                blocker = "not_reported"
            elif current >= required:
                advance_seconds = 0.0
            elif sealed:
                # A sealed source cannot be advanced any further: do not
                # suggest one, flag the plan unattainable instead.
                blocker = "sealed_below_required"
                shortfalls.append(
                    f"source {sid!r} is sealed at"
                    f" {format_event_time(parse_event_time(current))}, below the"
                    f" required effective max"
                    f" {format_event_time(parse_event_time(required))}"
                )
            else:
                advance_seconds = (
                    parse_event_time(required) - parse_event_time(current)
                ).total_seconds()

            sources.append(
                {
                    "source_id": sid,
                    "current": (
                        format_event_time(parse_event_time(current))
                        if current
                        else None
                    ),
                    "required": format_event_time(parse_event_time(required)),
                    "advance_seconds": advance_seconds,
                    "blocker": blocker,
                }
            )

        attainable = not shortfalls
        reason = None
        if not attainable:
            reason = (
                "release plan is unattainable: "
                + "; ".join(shortfalls)
                + "; sealed sources cannot be advanced"
            )

        return {
            "session_id": session_id,
            "count": count,
            "buffered": buffered,
            "allowed_lateness_seconds": lateness,
            "target_watermark": format_event_time(parse_event_time(target)),
            "attainable": attainable,
            "reason": reason,
            "sources": sources,
        }
