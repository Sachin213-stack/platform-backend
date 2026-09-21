"""Add log_entries table with RLS tenant isolation

Revision ID: 0005_add_log_entries_table
Revises: 0004_avatar_and_org_fields
Create Date: 2026-09-21 17:45:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "0005_add_log_entries_table"
down_revision: Union[str, None] = "0004_avatar_and_org_fields"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create log_entries table
    op.create_table(
        "log_entries",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("business_id", postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column("log_type", sa.String(length=20), server_default="application", nullable=False),
        sa.Column("source", sa.String(length=512), nullable=True),
        sa.Column("format", sa.String(length=20), server_default="text", nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("parsed_fields", sa.JSON(), server_default="{}", nullable=True),
        sa.Column("level", sa.String(length=20), server_default="info", nullable=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("ingested_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # Indexes
    op.create_index("idx_log_entries_biz_time", "log_entries", ["business_id", "timestamp"])
    op.create_index("idx_log_entries_biz_level_time", "log_entries", ["business_id", "level", "timestamp"])
    op.create_index("idx_log_entries_biz_type_time", "log_entries", ["business_id", "log_type", "timestamp"])
    op.create_index("idx_log_entries_biz_source_time", "log_entries", ["business_id", "source", "timestamp"])

    # PostgreSQL Row-Level Security (RLS) Policy
    op.execute("ALTER TABLE log_entries ENABLE ROW LEVEL SECURITY;")
    op.execute("ALTER TABLE log_entries FORCE ROW LEVEL SECURITY;")
    op.execute(
        """
        CREATE POLICY log_entries_tenant_isolation_policy ON log_entries
        FOR ALL
        USING (
            business_id = NULLIF(current_setting('app.current_business_id', true), '')::uuid
        )
        WITH CHECK (
            business_id = NULLIF(current_setting('app.current_business_id', true), '')::uuid
        );
        """
    )


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS log_entries_tenant_isolation_policy ON log_entries;")
    op.drop_index("idx_log_entries_biz_source_time", table_name="log_entries")
    op.drop_index("idx_log_entries_biz_type_time", table_name="log_entries")
    op.drop_index("idx_log_entries_biz_level_time", table_name="log_entries")
    op.drop_index("idx_log_entries_biz_time", table_name="log_entries")
    op.drop_table("log_entries")
