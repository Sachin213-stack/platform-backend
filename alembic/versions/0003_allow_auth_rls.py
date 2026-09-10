"""Update RLS policies for auth and registration

Revision ID: 0003_allow_auth_rls
Revises: 0002_enable_timescaledb
Create Date: 2026-09-10 12:30:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0003_allow_auth_rls"
down_revision: Union[str, None] = "0002_enable_timescaledb"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop strict RLS policies on users and businesses
    op.execute("DROP POLICY IF EXISTS users_tenant_isolation_policy ON users;")
    op.execute("DROP POLICY IF EXISTS businesses_tenant_isolation_policy ON businesses;")

    # Recreate policies permitting unscoped auth lookups / new business registration when session RLS is unset
    op.execute(
        """
        CREATE POLICY users_tenant_isolation_policy ON users
        FOR ALL
        USING (
            NULLIF(current_setting('app.current_business_id', true), '') IS NULL
            OR business_id = NULLIF(current_setting('app.current_business_id', true), '')::uuid
        )
        WITH CHECK (
            NULLIF(current_setting('app.current_business_id', true), '') IS NULL
            OR business_id = NULLIF(current_setting('app.current_business_id', true), '')::uuid
        );
        """
    )

    op.execute(
        """
        CREATE POLICY businesses_tenant_isolation_policy ON businesses
        FOR ALL
        USING (
            NULLIF(current_setting('app.current_business_id', true), '') IS NULL
            OR id = NULLIF(current_setting('app.current_business_id', true), '')::uuid
        )
        WITH CHECK (
            NULLIF(current_setting('app.current_business_id', true), '') IS NULL
            OR id = NULLIF(current_setting('app.current_business_id', true), '')::uuid
        );
        """
    )


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS users_tenant_isolation_policy ON users;")
    op.execute("DROP POLICY IF EXISTS businesses_tenant_isolation_policy ON businesses;")

    op.execute(
        """
        CREATE POLICY users_tenant_isolation_policy ON users
        FOR ALL
        USING (
            business_id = NULLIF(current_setting('app.current_business_id', true), '')::uuid
        )
        WITH CHECK (
            business_id = NULLIF(current_setting('app.current_business_id', true), '')::uuid
        );
        """
    )

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
