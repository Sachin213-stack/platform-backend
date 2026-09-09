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
    # Enable TimescaleDB extension on PostgreSQL
    # Required for Render Managed PostgreSQL instances where the extension
    # is available but must be enabled per database.
    # -------------------------------------------------------------
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")


def downgrade() -> None:
    op.execute("DROP EXTENSION IF EXISTS timescaledb CASCADE;")
