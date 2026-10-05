"""Forecast source: live run or simulated backtest week

Revision ID: 007
Revises: 006
Create Date: 2026-10-05 18:00:00.000000

Adds a nullable ``source`` column to ``forecast_snapshots``. Snapshots are
immutable (UPDATE is rejected by trigger and covered by the snapshot hash), so
existing rows are not backfilled: they keep NULL and their source is inferred
when read (``schemas.snapshot.resolve_forecast_source``).
"""
from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '007'
down_revision: Union[str, None] = '006'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('forecast_snapshots', sa.Column('source', sa.String(10), nullable=True))


def downgrade() -> None:
    op.drop_column('forecast_snapshots', 'source')
