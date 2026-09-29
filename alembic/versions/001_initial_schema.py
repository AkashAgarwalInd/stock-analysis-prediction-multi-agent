"""Initial schema

Revision ID: 001
Revises: 
Create Date: 2024-01-01 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'symbols',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('symbol', sa.String(20), nullable=False, unique=True),
        sa.Column('name', sa.String(200), nullable=False),
        sa.Column('sector', sa.String(100), nullable=True),
        sa.Column('industry', sa.String(100), nullable=True),
        sa.Column('exchange', sa.String(20), nullable=False, default='NSE'),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
    )
    op.create_index('ix_symbols_symbol', 'symbols', ['symbol'], unique=True)
    op.create_index('ix_symbols_sector', 'symbols', ['sector'])

    op.create_table(
        'forecasts',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('symbol_id', sa.Integer(), nullable=False),
        sa.Column('forecast_date', sa.Date(), nullable=False),
        sa.Column('horizon_days', sa.Integer(), nullable=False, default=5),
        sa.Column('baseline_forecast_json', sa.Text(), nullable=False),
        sa.Column('llm_forecast_json', sa.Text(), nullable=True),
        sa.Column('final_forecast_json', sa.Text(), nullable=False),
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.Column('status', sa.String(20), nullable=False, default='pending'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.ForeignKeyConstraint(['symbol_id'], ['symbols.id'], ondelete='CASCADE'),
    )
    op.create_index('ix_forecasts_symbol_date', 'forecasts', ['symbol_id', 'forecast_date'], unique=True)
    op.create_index('ix_forecasts_status', 'forecasts', ['status'])

    op.create_table(
        'forecast_evaluations',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('forecast_id', sa.Integer(), nullable=False),
        sa.Column('actual_prices_json', sa.Text(), nullable=False),
        sa.Column('baseline_metrics_json', sa.Text(), nullable=False),
        sa.Column('llm_metrics_json', sa.Text(), nullable=True),
        sa.Column('final_metrics_json', sa.Text(), nullable=False),
        sa.Column('evaluation_date', sa.Date(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.ForeignKeyConstraint(['forecast_id'], ['forecasts.id'], ondelete='CASCADE'),
    )
    op.create_index('ix_forecast_evaluations_forecast', 'forecast_evaluations', ['forecast_id'], unique=True)

    op.create_table(
        'postmortems',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('forecast_id', sa.Integer(), nullable=False),
        sa.Column('analysis_json', sa.Text(), nullable=False),
        sa.Column('lessons_learned', sa.Text(), nullable=True),
        sa.Column('calibration_adjustment_json', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.ForeignKeyConstraint(['forecast_id'], ['forecasts.id'], ondelete='CASCADE'),
    )
    op.create_index('ix_postmortems_forecast', 'postmortems', ['forecast_id'], unique=True)

    op.create_table(
        'audit_log',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('entity_type', sa.String(50), nullable=False),
        sa.Column('entity_id', sa.Integer(), nullable=False),
        sa.Column('action', sa.String(50), nullable=False),
        sa.Column('changes_json', sa.Text(), nullable=True),
        sa.Column('user_id', sa.String(100), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
    )
    op.create_index('ix_audit_log_entity', 'audit_log', ['entity_type', 'entity_id'])
    op.create_index('ix_audit_log_created', 'audit_log', ['created_at'])


def downgrade() -> None:
    op.drop_index('ix_audit_log_created', table_name='audit_log')
    op.drop_index('ix_audit_log_entity', table_name='audit_log')
    op.drop_table('audit_log')
    op.drop_index('ix_postmortems_forecast', table_name='postmortems')
    op.drop_table('postmortems')
    op.drop_index('ix_forecast_evaluations_forecast', table_name='forecast_evaluations')
    op.drop_table('forecast_evaluations')
    op.drop_index('ix_forecasts_status', table_name='forecasts')
    op.drop_index('ix_forecasts_symbol_date', table_name='forecasts')
    op.drop_table('forecasts')
    op.drop_index('ix_symbols_sector', table_name='symbols')
    op.drop_index('ix_symbols_symbol', table_name='symbols')
    op.drop_table('symbols')