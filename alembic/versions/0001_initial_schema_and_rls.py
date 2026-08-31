"""Initial schema and PostgreSQL Row-Level Security (RLS) policies

Revision ID: 0001_initial_schema_and_rls
Revises: 
Create Date: 2026-08-31 22:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0001_initial_schema_and_rls"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # -------------------------------------------------------------
    # 1. Businesses Table
    # -------------------------------------------------------------
    op.create_table(
        "businesses",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("slug", sa.String(255), unique=True, index=True, nullable=False),
        sa.Column("plan_tier", sa.String(50), server_default="starter", nullable=False),
        sa.Column("retention_days", sa.Integer(), server_default="30", nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("settings_config", sa.JSON(), server_default="{}", nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # -------------------------------------------------------------
    # 2. Users Table
    # -------------------------------------------------------------
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("business_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("businesses.id", ondelete="CASCADE"), index=True, nullable=False),
        sa.Column("email", sa.String(255), unique=True, index=True, nullable=False),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("full_name", sa.String(255), nullable=True),
        sa.Column("role", sa.String(50), server_default="member", nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # -------------------------------------------------------------
    # 3. API Keys Table
    # -------------------------------------------------------------
    op.create_table(
        "api_keys",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("business_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("businesses.id", ondelete="CASCADE"), index=True, nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("key_prefix", sa.String(16), index=True, nullable=False),
        sa.Column("encrypted_secret", sa.String(512), nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # -------------------------------------------------------------
    # 4. Telemetry Events Table
    # -------------------------------------------------------------
    op.create_table(
        "telemetry_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("business_id", postgresql.UUID(as_uuid=True), index=True, nullable=False),
        sa.Column("idempotency_key", sa.String(64), unique=True, index=True, nullable=True),
        sa.Column("event_type", sa.String(100), index=True, nullable=False),
        sa.Column("response_time_ms", sa.Float(), nullable=True),
        sa.Column("status_code", sa.Integer(), nullable=True),
        sa.Column("orders_count", sa.Integer(), server_default="0", nullable=True),
        sa.Column("revenue_amount", sa.Float(), server_default="0.0", nullable=True),
        sa.Column("cpu_usage_pct", sa.Float(), nullable=True),
        sa.Column("memory_usage_pct", sa.Float(), nullable=True),
        sa.Column("queue_depth", sa.Integer(), nullable=True),
        sa.Column("endpoint", sa.String(512), nullable=True),
        sa.Column("payload_metadata", sa.JSON(), server_default="{}", nullable=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=sa.func.now(), index=True, nullable=False),
    )
    op.create_index("idx_telemetry_biz_time", "telemetry_events", ["business_id", "timestamp"])
    op.create_index("idx_telemetry_biz_type_time", "telemetry_events", ["business_id", "event_type", "timestamp"])

    # -------------------------------------------------------------
    # 5. Anomalies Table (ML)
    # -------------------------------------------------------------
    op.create_table(
        "anomalies",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("business_id", postgresql.UUID(as_uuid=True), index=True, nullable=False),
        sa.Column("metric_name", sa.String(100), nullable=False),
        sa.Column("severity", sa.String(20), server_default="medium", nullable=False),
        sa.Column("expected_value", sa.Float(), nullable=False),
        sa.Column("actual_value", sa.Float(), nullable=False),
        sa.Column("confidence_score", sa.Float(), server_default="0.0", nullable=False),
        sa.Column("description", sa.String(500), nullable=False),
        sa.Column("is_resolved", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("detected_at", sa.DateTime(timezone=True), server_default=sa.func.now(), index=True, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("idx_anomalies_biz_detected", "anomalies", ["business_id", "detected_at"])

    # -------------------------------------------------------------
    # 6. Forecasts Table (ML)
    # -------------------------------------------------------------
    op.create_table(
        "forecasts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("business_id", postgresql.UUID(as_uuid=True), index=True, nullable=False),
        sa.Column("metric_name", sa.String(100), nullable=False),
        sa.Column("forecast_horizon", sa.String(50), server_default="24h", nullable=False),
        sa.Column("crash_risk_pct", sa.Float(), server_default="0.0", nullable=True),
        sa.Column("forecast_curve", sa.JSON(), server_default="{}", nullable=False),
        sa.Column("generated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), index=True, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("idx_forecasts_biz_generated", "forecasts", ["business_id", "generated_at"])

    # -------------------------------------------------------------
    # 7. Alert Rules Table
    # -------------------------------------------------------------
    op.create_table(
        "alert_rules",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("business_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("businesses.id", ondelete="CASCADE"), index=True, nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("metric_target", sa.String(100), nullable=False),
        sa.Column("threshold_value", sa.Float(), nullable=False),
        sa.Column("condition", sa.String(20), server_default="gt", nullable=False),
        sa.Column("channel", sa.String(50), server_default="email", nullable=False),
        sa.Column("channel_config", sa.JSON(), server_default="{}", nullable=True),
        sa.Column("is_enabled", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # -------------------------------------------------------------
    # 8. Conversations Table (FRIDAY)
    # -------------------------------------------------------------
    op.create_table(
        "conversations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("business_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("businesses.id", ondelete="CASCADE"), index=True, nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("title", sa.String(255), server_default="New Conversation", nullable=False),
        sa.Column("mode", sa.String(20), server_default="chat", nullable=False),
        sa.Column("messages", sa.JSON(), server_default="[]", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # -------------------------------------------------------------
    # 9. Auto-update `updated_at` Trigger
    # SQLAlchemy's `onupdate` only works via ORM — this trigger
    # ensures direct SQL UPDATEs also refresh the timestamp.
    # -------------------------------------------------------------
    op.execute(
        """
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """
    )

    # Apply trigger to all tables that have updated_at
    tables_with_updated_at = [
        "businesses", "users", "api_keys",
        "anomalies", "forecasts", "alert_rules", "conversations",
    ]
    for table in tables_with_updated_at:
        op.execute(
            f"""
            CREATE TRIGGER trg_{table}_updated_at
            BEFORE UPDATE ON {table}
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
            """
        )

    # -------------------------------------------------------------
    # 10. PostgreSQL Row-Level Security (RLS) Policies
    #
    # IMPORTANT: RLS is enforced via FORCE ROW LEVEL SECURITY, but
    # PostgreSQL superusers (e.g. 'postgres') BYPASS RLS by default.
    # For production, the app MUST connect as a non-superuser role,
    # or use SET ROLE to downgrade privileges after connecting.
    # -------------------------------------------------------------
    # NOTE: Table names below are hardcoded constants — f-string interpolation
    # is safe here. Do NOT use dynamic/user-provided table names in this pattern.
    tenant_tables = [
        "telemetry_events",
        "anomalies",
        "forecasts",
        "alert_rules",
        "conversations",
        "api_keys",
        "users",
    ]

    for table in tenant_tables:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY;")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY;")
        op.execute(
            f"""
            CREATE POLICY {table}_tenant_isolation_policy ON {table}
            FOR ALL
            USING (
                business_id = NULLIF(current_setting('app.current_business_id', true), '')::uuid
            )
            WITH CHECK (
                business_id = NULLIF(current_setting('app.current_business_id', true), '')::uuid
            );
            """
        )

    # Businesses table self-isolation policy
    op.execute("ALTER TABLE businesses ENABLE ROW LEVEL SECURITY;")
    op.execute("ALTER TABLE businesses FORCE ROW LEVEL SECURITY;")
    op.execute(
        """
        CREATE POLICY businesses_tenant_isolation_policy ON businesses
        FOR ALL
        USING (
            id = NULLIF(current_setting('app.current_business_id', true), '')::uuid
        )
        WITH CHECK (
            id = NULLIF(current_setting('app.current_business_id', true), '')::uuid
        );
        """
    )


def downgrade() -> None:
    # Drop updated_at triggers
    tables_with_updated_at = [
        "businesses", "users", "api_keys",
        "anomalies", "forecasts", "alert_rules", "conversations",
    ]
    for table in tables_with_updated_at:
        op.execute(f"DROP TRIGGER IF EXISTS trg_{table}_updated_at ON {table};")
    op.execute("DROP FUNCTION IF EXISTS update_updated_at_column();")

    # Drop RLS Policies
    tables = [
        "businesses",
        "users",
        "api_keys",
        "conversations",
        "alert_rules",
        "forecasts",
        "anomalies",
        "telemetry_events",
    ]
    for table in tables:
        op.execute(f"DROP POLICY IF EXISTS {table}_tenant_isolation_policy ON {table};")
        op.execute(f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY;")
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY;")

    # Drop Tables in Reverse Dependency Order
    op.drop_table("conversations")
    op.drop_table("alert_rules")
    op.drop_table("forecasts")
    op.drop_table("anomalies")
    op.drop_table("telemetry_events")
    op.drop_table("api_keys")
    op.drop_table("users")
    op.drop_table("businesses")
