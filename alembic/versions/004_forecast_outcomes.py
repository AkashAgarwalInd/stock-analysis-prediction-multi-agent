"""Forecast outcomes: actual-vs-forecast evaluation

Revision ID: 004
Revises: 003
Create Date: 2026-10-03 00:00:00.000000

Only final results (scored or invalid) are stored; an unresolved evaluation
(data not yet available) is retried on the next review. Both tables are
append-only, like the snapshots they evaluate.
"""
from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '004'
down_revision: Union[str, None] = '003'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

IMMUTABLE_TABLES = ('forecast_outcomes', 'outcome_daily')


def upgrade() -> None:
    op.create_table(
        'forecast_outcomes',
        sa.Column('forecast_id', sa.String(64), primary_key=True),
        sa.Column('status', sa.String(12), nullable=False),
        sa.Column('invalid_reason', sa.Text(), nullable=True),
        sa.Column('evaluated_at', sa.String(40), nullable=False),
        sa.Column('scorer_version', sa.String(10), nullable=False),
        sa.Column('as_of_date', sa.String(10), nullable=False),
        sa.Column('target_date', sa.String(10), nullable=False),
        sa.Column('validity_checks', sa.Text(), nullable=False),
        sa.Column('corporate_actions', sa.Text(), nullable=False),
        sa.Column('adjustment_factor', sa.Float(), nullable=True),
        sa.Column('actual_close', sa.Float(), nullable=True),
        sa.Column('actual_close_on_forecast_basis', sa.Float(), nullable=True),
        sa.Column('actual_return_pct', sa.Float(), nullable=True),
        sa.Column('realized_direction', sa.String(5), nullable=True),
        sa.Column('predicted_direction', sa.String(5), nullable=True),
        sa.Column('direction_correct', sa.Boolean(), nullable=True),
        sa.Column('signed_error_pct', sa.Float(), nullable=True),
        sa.Column('abs_error_pct', sa.Float(), nullable=True),
        sa.Column('in_80pct_band', sa.Boolean(), nullable=True),
        sa.Column('daily_band_breaches', sa.Integer(), nullable=True),
        sa.Column('pit_percentile', sa.Float(), nullable=True),
        sa.Column('brier', sa.Float(), nullable=True),
        sa.Column('pinball', sa.Text(), nullable=True),
        sa.Column('realized_vol_pct', sa.Float(), nullable=True),
        sa.Column('vol_ratio', sa.Float(), nullable=True),
        sa.Column('nifty_return_pct', sa.Float(), nullable=True),
        sa.Column('excess_vs_nifty_pct', sa.Float(), nullable=True),
        sa.Column('beta_adjusted_excess_pct', sa.Float(), nullable=True),
        sa.Column('loss_metric', sa.String(20), nullable=False),
        sa.Column('baseline_signed_error_pct', sa.Float(), nullable=True),
        sa.Column('baseline_brier', sa.Float(), nullable=True),
        sa.Column('baseline_pinball', sa.Text(), nullable=True),
        sa.Column('baseline_loss', sa.Float(), nullable=True),
        sa.Column('final_loss', sa.Float(), nullable=True),
        sa.Column('llm_value_added', sa.Float(), nullable=True),
        sa.Column('llm_value_added_pinball', sa.Float(), nullable=True),
        sa.Column('analyst_hits', sa.Text(), nullable=True),
        sa.Column('decision_engine', sa.String(60), nullable=True),
        sa.Column('decision_engine_enabled', sa.Boolean(), nullable=False),
        sa.Column('market_regime_decision', sa.String(30), nullable=True),
        sa.Column('adjustment_gate_decision', sa.String(30), nullable=True),
        sa.Column('adjustment_applied', sa.Boolean(), nullable=False),
        sa.Column('fallback_to_quant', sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(['forecast_id'], ['forecast_snapshots.forecast_id']),
        sa.CheckConstraint("status IN ('scored', 'invalid')", name='ck_forecast_outcomes_status'),
    )
    op.create_index('ix_forecast_outcomes_status', 'forecast_outcomes', ['status', 'target_date'])

    op.create_table(
        'outcome_daily',
        sa.Column('forecast_id', sa.String(64), nullable=False),
        sa.Column('date', sa.String(10), nullable=False),
        sa.Column('close', sa.Float(), nullable=False),
        sa.Column('close_on_forecast_basis', sa.Float(), nullable=False),
        sa.Column('nifty_close', sa.Float(), nullable=True),
        sa.Column('in_daily_band', sa.Boolean(), nullable=True),
        sa.ForeignKeyConstraint(['forecast_id'], ['forecast_outcomes.forecast_id']),
        sa.PrimaryKeyConstraint('forecast_id', 'date'),
    )

    for table in IMMUTABLE_TABLES:
        for action in ('UPDATE', 'DELETE'):
            op.execute(
                f"CREATE TRIGGER {table}_no_{action.lower()} BEFORE {action} ON {table} "
                f"BEGIN SELECT RAISE(ABORT, '{table} is immutable: outcomes are final'); END"
            )


def downgrade() -> None:
    for table in IMMUTABLE_TABLES:
        for action in ('update', 'delete'):
            op.execute(f"DROP TRIGGER IF EXISTS {table}_no_{action}")
    op.drop_table('outcome_daily')
    op.drop_index('ix_forecast_outcomes_status', table_name='forecast_outcomes')
    op.drop_table('forecast_outcomes')
