"""Immutable forecast snapshots, decision audit and daily predictions

Revision ID: 002
Revises: 001
Create Date: 2026-10-02 00:00:00.000000

The legacy ``forecasts`` table from 001 (mutable ``status``, one row per
symbol/day) is left in place and unused; snapshots live in their own tables.
UPDATE and DELETE are rejected by triggers so a finalized forecast can only be
corrected by inserting a new version.
"""
from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '002'
down_revision: Union[str, None] = '001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

IMMUTABLE_TABLES = (
    'forecast_snapshots',
    'forecast_decisions',
    'forecast_daily_predictions',
    'price_history_snapshots',
)


def upgrade() -> None:
    # Content-addressed close series behind each quant baseline, so a forecast can be
    # reproduced even after the data vendor revises history.
    op.create_table(
        'price_history_snapshots',
        sa.Column('history_sha256', sa.String(64), primary_key=True),
        sa.Column('symbol', sa.String(30), nullable=False),
        sa.Column('bars', sa.Integer(), nullable=False),
        sa.Column('first_date', sa.String(10), nullable=True),
        sa.Column('last_date', sa.String(10), nullable=True),
        sa.Column('dates_json', sa.Text(), nullable=False),
        sa.Column('closes_json', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
    )

    op.create_table(
        'forecast_snapshots',
        sa.Column('forecast_id', sa.String(64), primary_key=True),
        sa.Column('root_forecast_id', sa.String(64), nullable=False),
        sa.Column('version', sa.Integer(), nullable=False),
        sa.Column('supersedes_forecast_id', sa.String(64), nullable=True),
        sa.Column('correction_reason', sa.Text(), nullable=True),
        sa.Column('corrected_at', sa.String(40), nullable=True),
        sa.Column('ticker', sa.String(20), nullable=False),
        sa.Column('resolved_symbol', sa.String(30), nullable=False),
        sa.Column('company_name', sa.String(200), nullable=False),
        sa.Column('made_at', sa.String(40), nullable=False),
        sa.Column('as_of_date', sa.Date(), nullable=False),
        sa.Column('target_date', sa.Date(), nullable=False),
        sa.Column('horizon_trading_days', sa.Integer(), nullable=False),
        sa.Column('last_close', sa.Float(), nullable=False),
        sa.Column('prob_up', sa.Float(), nullable=False),
        sa.Column('prob_flat', sa.Float(), nullable=False),
        sa.Column('prob_down', sa.Float(), nullable=False),
        sa.Column('expected_return_pct', sa.Float(), nullable=False),
        sa.Column('p10_price', sa.Float(), nullable=False),
        sa.Column('p50_price', sa.Float(), nullable=False),
        sa.Column('p90_price', sa.Float(), nullable=False),
        sa.Column('adjustment_applied', sa.Boolean(), nullable=False),
        sa.Column('fallback_to_quant', sa.Boolean(), nullable=False),
        sa.Column('market_regime', sa.String(30), nullable=False),
        sa.Column('risk_category', sa.String(20), nullable=False),
        sa.Column('data_snapshot_id', sa.String(64), nullable=False),
        sa.Column('calibration_version', sa.Integer(), nullable=False),
        sa.Column('model_versions_json', sa.Text(), nullable=False),
        sa.Column('prompt_versions_json', sa.Text(), nullable=False),
        sa.Column('quant_baseline_json', sa.Text(), nullable=False),
        sa.Column('final_forecast_json', sa.Text(), nullable=False),
        sa.Column('analyst_reports_json', sa.Text(), nullable=False),
        sa.Column('critic_json', sa.Text(), nullable=True),
        sa.Column('data_inputs_json', sa.Text(), nullable=False),
        sa.Column('data_quality_json', sa.Text(), nullable=False),
        sa.Column('snapshot_hash', sa.String(64), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.ForeignKeyConstraint(['supersedes_forecast_id'], ['forecast_snapshots.forecast_id']),
        sa.UniqueConstraint('root_forecast_id', 'version', name='uq_forecast_snapshots_root_version'),
        sa.CheckConstraint('version >= 1', name='ck_forecast_snapshots_version'),
    )
    op.create_index('ix_forecast_snapshots_ticker_as_of', 'forecast_snapshots', ['ticker', 'as_of_date'])
    op.create_index('ix_forecast_snapshots_target_date', 'forecast_snapshots', ['target_date'])
    op.create_index('ix_forecast_snapshots_data_snapshot', 'forecast_snapshots', ['data_snapshot_id'])

    op.create_table(
        'forecast_decisions',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('forecast_id', sa.String(64), nullable=False),
        sa.Column('sequence', sa.Integer(), nullable=False),
        sa.Column('decision_engine', sa.String(50), nullable=False),
        sa.Column('decision_model', sa.String(50), nullable=False),
        sa.Column('decision_model_version', sa.String(30), nullable=False),
        sa.Column('decision_type', sa.String(40), nullable=False),
        sa.Column('decision', sa.String(40), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('rationale', sa.Text(), nullable=False),
        sa.Column('evidence_json', sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(['forecast_id'], ['forecast_snapshots.forecast_id']),
        sa.UniqueConstraint('forecast_id', 'sequence', name='uq_forecast_decisions_sequence'),
    )
    op.create_index('ix_forecast_decisions_type', 'forecast_decisions', ['decision_type', 'decision'])

    op.create_table(
        'forecast_daily_predictions',
        sa.Column('forecast_id', sa.String(64), nullable=False),
        sa.Column('target_date', sa.Date(), nullable=False),
        sa.Column('day_index', sa.Integer(), nullable=False),
        sa.Column('predicted_return_pct', sa.Float(), nullable=False),
        sa.Column('p10_price', sa.Float(), nullable=False),
        sa.Column('p50_price', sa.Float(), nullable=False),
        sa.Column('p90_price', sa.Float(), nullable=False),
        sa.Column('prob_up', sa.Float(), nullable=False),
        sa.Column('source', sa.String(30), nullable=False),
        sa.ForeignKeyConstraint(['forecast_id'], ['forecast_snapshots.forecast_id']),
        sa.PrimaryKeyConstraint('forecast_id', 'target_date'),
    )

    for table in IMMUTABLE_TABLES:
        for action in ('UPDATE', 'DELETE'):
            op.execute(
                f"CREATE TRIGGER {table}_no_{action.lower()} BEFORE {action} ON {table} "
                f"BEGIN SELECT RAISE(ABORT, '{table} is immutable: record a new version instead'); END"
            )


def downgrade() -> None:
    for table in IMMUTABLE_TABLES:
        for action in ('update', 'delete'):
            op.execute(f"DROP TRIGGER IF EXISTS {table}_no_{action}")
    op.drop_table('forecast_daily_predictions')
    op.drop_index('ix_forecast_decisions_type', table_name='forecast_decisions')
    op.drop_table('forecast_decisions')
    op.drop_index('ix_forecast_snapshots_data_snapshot', table_name='forecast_snapshots')
    op.drop_index('ix_forecast_snapshots_target_date', table_name='forecast_snapshots')
    op.drop_index('ix_forecast_snapshots_ticker_as_of', table_name='forecast_snapshots')
    op.drop_table('forecast_snapshots')
    op.drop_table('price_history_snapshots')
