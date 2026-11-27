"""add confidence to chat_messages

Revision ID: d2a9f47c1b63
Revises: e5b9c2d71f40
Create Date: 2026-11-27

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd2a9f47c1b63'
down_revision: Union[str, Sequence[str], None] = 'e5b9c2d71f40'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('chat_messages', sa.Column('confidence', sa.Float(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('chat_messages', 'confidence')
