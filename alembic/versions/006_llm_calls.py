"""LLM call log: tokens, latency, estimated cost per call (Plan.md §37.16, §50-51)

Revision ID: 006
Revises: 005
Create Date: 2026-10-05 12:00:00.000000

Operational telemetry, not evaluation evidence: rows are only appended, except
that ``forecast_id`` is filled in once the run's forecast has been stored.
"""
from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '006'
down_revision: Union[str, None] = '005'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'llm_calls',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('run_id', sa.String(64), nullable=True),
        sa.Column('forecast_id', sa.String(64), nullable=True),
        sa.Column('ticker', sa.String(20), nullable=True),
        sa.Column('node', sa.String(60), nullable=True),
        sa.Column('role', sa.String(20), nullable=False),
        sa.Column('model', sa.String(100), nullable=False),
        sa.Column('purpose', sa.String(20), nullable=False),
        sa.Column('prompt_tokens', sa.Integer(), nullable=False),
        sa.Column('completion_tokens', sa.Integer(), nullable=False),
        sa.Column('latency_ms', sa.Integer(), nullable=False),
        sa.Column('cost_est', sa.Float(), nullable=False),
        sa.Column('status', sa.String(10), nullable=False),
        sa.Column('attempts', sa.Integer(), nullable=False),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('created_at', sa.String(40), nullable=False),
    )
    op.create_index('ix_llm_calls_run_id', 'llm_calls', ['run_id'])
    op.create_index('ix_llm_calls_created_at', 'llm_calls', ['created_at'])


def downgrade() -> None:
    op.drop_index('ix_llm_calls_created_at', table_name='llm_calls')
    op.drop_index('ix_llm_calls_run_id', table_name='llm_calls')
    op.drop_table('llm_calls')
