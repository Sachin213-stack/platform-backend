"""Enable TimescaleDB extension

Revision ID: 0002_enable_timescaledb
Revises: 0001_initial_schema_and_rls
Create Date: 2026-09-09 23:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0002_enable_timescaledb"
down_revision: Union[str, None] = "0001_initial_schema_and_rls"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # -------------------------------------------------------------
    # Enable TimescaleDB extension on PostgreSQL if supported
    # -------------------------------------------------------------
    op.execute(
        """
        DO $$
        BEGIN
            CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
        EXCEPTION WHEN OTHERS THEN
            RAISE NOTICE 'timescaledb extension unavailable or skipped: %', SQLERRM;
        END $$;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DO $$
        BEGIN
            DROP EXTENSION IF EXISTS timescaledb CASCADE;
        EXCEPTION WHEN OTHERS THEN
            NULL;
        END $$;
        """
    )
