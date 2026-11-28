"""add verified_citations_count to ai_query_logs

Revision ID: f4b8c21d9a37
Revises: d2a9f47c1b63
Create Date: 2026-11-28

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f4b8c21d9a37'
down_revision: Union[str, Sequence[str], None] = 'd2a9f47c1b63'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('ai_query_logs', sa.Column('verified_citations_count', sa.Integer(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('ai_query_logs', 'verified_citations_count')
