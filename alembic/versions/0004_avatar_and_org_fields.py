"""Add avatar columns to users and org fields to businesses

Revision ID: 0004_avatar_and_org_fields
Revises: 0003_allow_auth_rls
Create Date: 2026-09-11 18:45:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0004_avatar_and_org_fields"
down_revision: Union[str, None] = "0003_allow_auth_rls"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add avatar binary data and mime-type to users
    op.add_column("users", sa.Column("avatar_data", sa.LargeBinary(), nullable=True))
    op.add_column("users", sa.Column("avatar_mime_type", sa.String(length=50), nullable=True))

    # Add business_type and ops_email to businesses
    op.add_column(
        "businesses",
        sa.Column("business_type", sa.String(length=50), server_default="ecommerce", nullable=False),
    )
    op.add_column("businesses", sa.Column("ops_email", sa.String(length=255), nullable=True))


def downgrade() -> None:
    op.drop_column("businesses", "ops_email")
    op.drop_column("businesses", "business_type")
    op.drop_column("users", "avatar_mime_type")
    op.drop_column("users", "avatar_data")
