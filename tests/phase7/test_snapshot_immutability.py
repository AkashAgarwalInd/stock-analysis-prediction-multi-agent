"""Phase 7: snapshots are immutable; corrections are new versions."""

import json
import sqlite3
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from stock_analysis.database import (
    ForecastSnapshotError,
    SnapshotExistsError,
    SnapshotIntegrityError,
    SnapshotLineageError,
)
from stock_analysis.schemas.analyst_reports import (
    AdjustmentGateDecision,
    DecisionType,
    MarketRegime,
    RiskCategory,
    RulesDecisionEngine,
)
from stock_analysis.snapshots import build_forecast_snapshot
from tests.forecast_helpers import MADE_AT

CORRECTED_AT = datetime(2026, 9, 30, 9, 0, tzinfo=UTC)


class TestDatabaseImmutability:
    @pytest.mark.parametrize(
        "statement",
        [
            "UPDATE forecast_snapshots SET prob_up = 0.99",
            "UPDATE forecast_snapshots SET final_forecast_json = '{}'",
            "UPDATE forecast_snapshots SET analyst_reports_json = '{}'",
            "UPDATE forecast_snapshots SET calibration_version = 7",
            "UPDATE forecast_decisions SET decision = 'allow_adjustment'",
            "UPDATE forecast_daily_predictions SET p50_price = 1.0",
            "DELETE FROM forecast_snapshots",
            "DELETE FROM forecast_decisions",
            "DELETE FROM forecast_daily_predictions",
        ],
    )
    def test_rewrites_are_rejected(self, store, snapshot, migrated_db, statement):
        store.save(snapshot)
        with pytest.raises(sqlite3.DatabaseError, match="immutable"):
            migrated_db.execute(statement)
        assert store.get(snapshot.forecast_id) == snapshot

    def test_saving_same_forecast_twice_is_rejected(self, store, snapshot):
        store.save(snapshot)
        with pytest.raises(SnapshotExistsError):
            store.save(snapshot)
        assert store.get(snapshot.forecast_id) == snapshot

    def test_failed_save_leaves_no_partial_rows(self, store, snapshot, migrated_db):
        # model_copy skips validation: a duplicate daily row fails mid-transaction,
        # after the snapshot and decision rows were already inserted.
        dup = snapshot.model_copy(
            update={
                "forecast_id": "other",
                "root_forecast_id": "other",
                "daily_predictions": snapshot.daily_predictions[:1] * 2,
            }
        )
        with pytest.raises(ForecastSnapshotError) as excinfo:
            store.save(dup)
        assert not isinstance(excinfo.value, SnapshotExistsError)
        assert store.get("other") is None
        assert (
            migrated_db.fetchone(
                "SELECT COUNT(*) AS n FROM forecast_decisions WHERE forecast_id = 'other'"
            )["n"]
            == 0
        )

    def test_tampering_is_detected_on_load(self, store, snapshot, migrated_db):
        store.save(snapshot)
        migrated_db.execute("DROP TRIGGER forecast_snapshots_no_update")
        final = json.loads(
            migrated_db.fetchone("SELECT final_forecast_json FROM forecast_snapshots")[0]
        )
        final["prob_up"], final["prob_down"] = final["prob_down"], final["prob_up"]
        migrated_db.execute(
            "UPDATE forecast_snapshots SET final_forecast_json = ?", (json.dumps(final),)
        )
        with pytest.raises(SnapshotIntegrityError):
            store.get(snapshot.forecast_id)

    def test_tampered_decision_row_is_detected(self, store, snapshot, migrated_db):
        store.save(snapshot)
        migrated_db.execute("DROP TRIGGER forecast_decisions_no_update")
        migrated_db.execute(
            "UPDATE forecast_decisions SET rationale = 'rewritten' WHERE decision_type = ?",
            (DecisionType.FORECAST_ADJUSTMENT_GATE.value,),
        )
        with pytest.raises(SnapshotIntegrityError):
            store.get(snapshot.forecast_id)


class TestModelImmutability:
    def test_snapshot_model_is_frozen(self, snapshot):
        with pytest.raises(ValidationError):
            snapshot.calibration_version = 5

    def test_version_one_cannot_claim_to_supersede(self, snapshot):
        data = snapshot.model_dump()
        data["supersedes_forecast_id"] = "something"
        with pytest.raises(ValidationError, match="version 1"):
            type(snapshot).model_validate(data)

    def test_correction_cannot_rewrite_lineage(self, snapshot):
        with pytest.raises(ValueError, match="lineage"):
            snapshot.corrected(reason="x", corrected_at=CORRECTED_AT, version=1)

    def test_correction_cannot_change_made_at(self, snapshot):
        with pytest.raises(ValueError, match="lineage"):
            snapshot.corrected(reason="x", corrected_at=CORRECTED_AT, made_at=CORRECTED_AT)

    def test_naive_made_at_is_rejected(self, snapshot):
        data = snapshot.model_dump()
        data["made_at"] = data["made_at"].replace(tzinfo=None)
        with pytest.raises(ValidationError):
            type(snapshot).model_validate(data)

    def test_inconsistent_daily_path_is_rejected(self, snapshot):
        data = snapshot.model_dump()
        data["daily_predictions"] = list(reversed(data["daily_predictions"]))
        with pytest.raises(ValidationError, match="daily predictions"):
            type(snapshot).model_validate(data)


class TestCorrections:
    def test_correction_is_a_new_version(self, store, snapshot):
        store.save(snapshot)
        corrected = store.record_correction(
            snapshot.forecast_id,
            reason="Company name was misspelled",
            corrected_at=CORRECTED_AT,
            company_name="Reliance Industries Limited",
        )

        assert corrected.forecast_id != snapshot.forecast_id
        assert corrected.version == 2
        assert corrected.root_forecast_id == snapshot.forecast_id
        assert corrected.supersedes_forecast_id == snapshot.forecast_id
        assert corrected.company_name == "Reliance Industries Limited"
        assert corrected.made_at == snapshot.made_at
        assert corrected.corrected_at == CORRECTED_AT

        original = store.get(snapshot.forecast_id)
        assert original == snapshot
        assert original.company_name == "Reliance Industries Ltd"
        assert original.final_forecast == snapshot.final_forecast
        assert original.decisions == snapshot.decisions
        assert original.analyst_reports == snapshot.analyst_reports
        assert original.calibration_version == snapshot.calibration_version

        assert store.get_latest(snapshot.forecast_id) == corrected
        assert [s.version for s in store.list_versions(corrected.forecast_id)] == [1, 2]

    def test_correction_keeps_original_calibration_version(self, store, adjusted_state):
        snap = build_forecast_snapshot(adjusted_state, made_at=MADE_AT, calibration_version=1)
        store.save(snap)
        store.record_correction(
            snap.forecast_id,
            reason="Recalibrated",
            corrected_at=CORRECTED_AT,
            calibration_version=2,
        )
        assert store.get(snap.forecast_id).calibration_version == 1
        assert store.get_latest(snap.forecast_id).calibration_version == 2

    def test_correction_must_extend_latest_version(self, store, snapshot):
        store.save(snapshot)
        store.record_correction(snapshot.forecast_id, reason="first fix", corrected_at=CORRECTED_AT)
        stale = snapshot.corrected(reason="branch from v1", corrected_at=CORRECTED_AT)
        with pytest.raises(SnapshotLineageError):
            store.save(stale)
        assert len(store.list_versions(snapshot.forecast_id)) == 2

    def test_correction_of_unknown_forecast_fails(self, store):
        with pytest.raises(SnapshotLineageError):
            store.record_correction("missing", reason="nothing to fix")


class TestDecisionAudit:
    def test_decision_metadata_persisted(self, store, snapshot, migrated_db):
        store.save(snapshot)
        rows = migrated_db.fetchall(
            "SELECT * FROM forecast_decisions WHERE forecast_id = ? ORDER BY sequence",
            (snapshot.forecast_id,),
        )

        assert [r["decision_type"] for r in rows] == [
            DecisionType.MARKET_REGIME.value,
            DecisionType.RISK_CATEGORY.value,
            DecisionType.FORECAST_ADJUSTMENT_GATE.value,
        ]
        gate = rows[2]
        assert gate["decision_engine"] == "adjustment_gate"
        assert gate["decision_model"] == "rules"
        assert gate["decision_model_version"] == "0.1.0"
        assert gate["decision"] == AdjustmentGateDecision.ALLOW_ADJUSTMENT.value
        assert gate["confidence"] == 0.7
        assert "confident analysts" in gate["rationale"]
        evidence = json.loads(gate["evidence_json"])
        assert {a["analyst"] for a in evidence["analysts"]} == {
            "technical",
            "fundamental",
            "sentiment",
            "context",
        }
        assert evidence["quant_baseline_available"] is True

    def test_gate_decision_is_reconstructable(self, store, snapshot):
        store.save(snapshot)
        loaded = store.get(snapshot.forecast_id)
        gate = loaded.decision(DecisionType.FORECAST_ADJUSTMENT_GATE)

        replay = RulesDecisionEngine().decide(
            DecisionType.FORECAST_ADJUSTMENT_GATE,
            {
                "analyst_reports": [r for r in loaded.analyst_reports.values() if r],
                "quant_baseline": loaded.quant_baseline,
            },
        )
        assert replay.result == gate.decision
        assert replay.rationale == gate.rationale
        assert replay.model == gate.decision_model

    def test_closed_gate_records_guardrail_reason(self, store, guardrail_state):
        snap = build_forecast_snapshot(guardrail_state, made_at=MADE_AT)
        store.save(snap)
        loaded = store.get(snap.forecast_id)

        gate = loaded.decision(DecisionType.FORECAST_ADJUSTMENT_GATE)
        assert gate.decision == AdjustmentGateDecision.NO_ADJUSTMENT.value
        assert gate.decision_model == "guardrails"
        assert gate.evidence["guardrail_violations"]
        regime = loaded.decision(DecisionType.MARKET_REGIME)
        assert regime.decision_engine == "guardrails"
        assert loaded.market_regime == MarketRegime.MIXED
        assert loaded.risk_category == RiskCategory.HIGH
        assert loaded.data_quality["guardrail_violations"]
        assert loaded.final_forecast.adjustment_applied is False
