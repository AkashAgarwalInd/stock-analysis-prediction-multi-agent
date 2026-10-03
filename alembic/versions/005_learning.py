"""Learning: shadow forecasts, postmortems, calibration params, decision outcomes

Revision ID: 005
Revises: 004
Create Date: 2026-10-03 12:00:00.000000

Every new table is append-only. ``forecast_postmortems`` is named so it does
not collide with the unused ``postmortems`` table from revision 001.
``forecast_outcomes`` gains the shadow-forecast (uncalibrated baseline)
columns; existing rows keep NULL there and calibration version 0.
"""
from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '005'
down_revision: Union[str, None] = '004'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

IMMUTABLE_TABLES = ('shadow_forecasts', 'calibration_params', 'forecast_postmortems', 'decision_outcomes')

OUTCOME_COLUMNS = (
    ('baseline_in_80pct_band', sa.Boolean()),
    ('uncalibrated_signed_error_pct', sa.Float()),
    ('uncalibrated_in_80pct_band', sa.Boolean()),
    ('uncalibrated_brier', sa.Float()),
    ('uncalibrated_pinball', sa.Text()),
    ('uncalibrated_loss', sa.Float()),
    ('calibration_value_added', sa.Float()),
    ('calibration_value_added_pinball', sa.Float()),
)


def upgrade() -> None:
    op.create_table(
        'shadow_forecasts',
        sa.Column('forecast_id', sa.String(64), primary_key=True),
        sa.Column('calibration_version', sa.Integer(), nullable=False),
        sa.Column('calibration_json', sa.Text(), nullable=True),
        sa.Column('uncalibrated_baseline_json', sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(['forecast_id'], ['forecast_snapshots.forecast_id']),
    )

    op.create_table(
        'calibration_params',
        sa.Column('ticker', sa.String(20), nullable=False),
        sa.Column('version', sa.Integer(), nullable=False),
        sa.Column('vol_multiplier', sa.Float(), nullable=False),
        sa.Column('p50_bias_shift_pct', sa.Float(), nullable=False),
        sa.Column('previous_vol_multiplier', sa.Float(), nullable=False),
        sa.Column('previous_p50_bias_shift_pct', sa.Float(), nullable=False),
        sa.Column('n_samples', sa.Integer(), nullable=False),
        sa.Column('reason_json', sa.Text(), nullable=False),
        sa.Column('evidence_json', sa.Text(), nullable=False),
        sa.Column('calibrator_version', sa.String(10), nullable=False),
        sa.Column('created_at', sa.String(40), nullable=False),
        sa.PrimaryKeyConstraint('ticker', 'version'),
        sa.CheckConstraint('version >= 1', name='ck_calibration_params_version'),
    )

    op.create_table(
        'forecast_postmortems',
        sa.Column('forecast_id', sa.String(64), primary_key=True),
        sa.Column('ticker', sa.String(20), nullable=False),
        sa.Column('method', sa.String(25), nullable=False),
        sa.Column('primary_cause', sa.String(40), nullable=False),
        sa.Column('confidence', sa.String(10), nullable=False),
        sa.Column('expected_noise', sa.Boolean(), nullable=False),
        sa.Column('knowable_at_forecast_time_json', sa.Text(), nullable=False),
        sa.Column('hindsight_json', sa.Text(), nullable=False),
        sa.Column('lessons_json', sa.Text(), nullable=False),
        sa.Column('postmortem_json', sa.Text(), nullable=False),
        sa.Column('created_at', sa.String(40), nullable=False),
        sa.ForeignKeyConstraint(['forecast_id'], ['forecast_outcomes.forecast_id']),
    )

    op.create_table(
        'decision_outcomes',
        sa.Column('forecast_id', sa.String(64), nullable=False),
        sa.Column('decision_type', sa.String(40), nullable=False),
        sa.Column('ticker', sa.String(20), nullable=False),
        sa.Column('as_of_date', sa.String(10), nullable=False),
        sa.Column('target_date', sa.String(10), nullable=False),
        sa.Column('decision_engine', sa.String(60), nullable=False),
        sa.Column('decision_model', sa.String(60), nullable=False),
        sa.Column('decision_model_version', sa.String(30), nullable=False),
        sa.Column('decision', sa.String(40), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('adjustment_applied', sa.Boolean(), nullable=False),
        sa.Column('fallback_to_quant', sa.Boolean(), nullable=False),
        sa.Column('calibration_version', sa.Integer(), nullable=False),
        sa.Column('direction_correct', sa.Boolean(), nullable=True),
        sa.Column('in_80pct_band', sa.Boolean(), nullable=True),
        sa.Column('final_loss', sa.Float(), nullable=True),
        sa.Column('baseline_loss', sa.Float(), nullable=True),
        sa.Column('uncalibrated_loss', sa.Float(), nullable=True),
        sa.Column('llm_value_added', sa.Float(), nullable=True),
        sa.Column('llm_value_added_pinball', sa.Float(), nullable=True),
        sa.Column('calibration_value_added', sa.Float(), nullable=True),
        sa.Column('primary_cause', sa.String(40), nullable=True),
        sa.Column('recorded_at', sa.String(40), nullable=False),
        sa.PrimaryKeyConstraint('forecast_id', 'decision_type'),
        sa.ForeignKeyConstraint(['forecast_id'], ['forecast_outcomes.forecast_id']),
    )
    op.create_index('ix_decision_outcomes_decision', 'decision_outcomes', ['decision_type', 'decision'])

    for name, column_type in OUTCOME_COLUMNS:
        op.add_column('forecast_outcomes', sa.Column(name, column_type, nullable=True))
    op.add_column(
        'forecast_outcomes',
        sa.Column('calibration_version', sa.Integer(), nullable=False, server_default='0'),
    )

    for table in IMMUTABLE_TABLES:
        for action in ('UPDATE', 'DELETE'):
            op.execute(
                f"CREATE TRIGGER {table}_no_{action.lower()} BEFORE {action} ON {table} "
                f"BEGIN SELECT RAISE(ABORT, '{table} is immutable: append a new record instead'); END"
            )


def downgrade() -> None:
    for table in IMMUTABLE_TABLES:
        for action in ('update', 'delete'):
            op.execute(f"DROP TRIGGER IF EXISTS {table}_no_{action}")
    # SQLite >= 3.35 drops columns in place, keeping the outcome immutability triggers
    for name in ('calibration_version', *(n for n, _ in OUTCOME_COLUMNS)):
        op.execute(f"ALTER TABLE forecast_outcomes DROP COLUMN {name}")
    op.drop_index('ix_decision_outcomes_decision', table_name='decision_outcomes')
    op.drop_table('decision_outcomes')
    op.drop_table('forecast_postmortems')
    op.drop_table('calibration_params')
    op.drop_table('shadow_forecasts')
