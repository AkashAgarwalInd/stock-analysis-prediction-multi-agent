"""Long-term memory: forecast insights, lessons and lesson evidence

Revision ID: 003
Revises: 002
Create Date: 2026-10-02 00:00:00.000000

All three tables are append-only. A lesson's status, evidence counts and
last confirmation are derived from ``lesson_evidence`` rather than stored as
mutable counters, so every lifecycle change stays auditable and can be
queried point-in-time.
"""
from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '003'
down_revision: Union[str, None] = '002'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

IMMUTABLE_TABLES = ('memory_insights', 'lessons', 'lesson_evidence')


def upgrade() -> None:
    op.create_table(
        'memory_insights',
        sa.Column('forecast_id', sa.String(64), primary_key=True),
        sa.Column('ticker', sa.String(20), nullable=False),
        sa.Column('made_at', sa.String(40), nullable=False),
        sa.Column('insight_json', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.ForeignKeyConstraint(['forecast_id'], ['forecast_snapshots.forecast_id']),
    )
    op.create_index('ix_memory_insights_ticker_made_at', 'memory_insights', ['ticker', 'made_at'])

    op.create_table(
        'lessons',
        sa.Column('lesson_id', sa.String(64), primary_key=True),
        sa.Column('scope', sa.String(10), nullable=False),
        sa.Column('ticker', sa.String(20), nullable=True),
        sa.Column('sector', sa.String(100), nullable=True),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('category', sa.String(60), nullable=False),
        sa.Column('source_forecast_id', sa.String(64), nullable=False),
        sa.Column('first_seen', sa.String(40), nullable=False),
        sa.ForeignKeyConstraint(['source_forecast_id'], ['forecast_snapshots.forecast_id']),
        sa.CheckConstraint("scope IN ('ticker', 'sector', 'general')", name='ck_lessons_scope'),
    )
    op.create_index('ix_lessons_scope', 'lessons', ['scope', 'ticker', 'sector'])

    op.create_table(
        'lesson_evidence',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('lesson_id', sa.String(64), nullable=False),
        sa.Column('kind', sa.String(15), nullable=False),
        sa.Column('forecast_id', sa.String(64), nullable=True),
        sa.Column('note', sa.Text(), nullable=True),
        sa.Column('recorded_at', sa.String(40), nullable=False),
        sa.ForeignKeyConstraint(['lesson_id'], ['lessons.lesson_id']),
        sa.ForeignKeyConstraint(['forecast_id'], ['forecast_snapshots.forecast_id']),
        sa.CheckConstraint(
            "kind IN ('confirmed', 'contradicted', 'retired')", name='ck_lesson_evidence_kind'
        ),
    )
    op.create_index('ix_lesson_evidence_lesson', 'lesson_evidence', ['lesson_id', 'recorded_at'])

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
    op.drop_index('ix_lesson_evidence_lesson', table_name='lesson_evidence')
    op.drop_table('lesson_evidence')
    op.drop_index('ix_lessons_scope', table_name='lessons')
    op.drop_table('lessons')
    op.drop_index('ix_memory_insights_ticker_made_at', table_name='memory_insights')
    op.drop_table('memory_insights')
