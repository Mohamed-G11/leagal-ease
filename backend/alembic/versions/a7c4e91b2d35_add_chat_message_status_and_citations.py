"""add status, error_code and citations to chat_messages

Revision ID: a7c4e91b2d35
Revises: 840b2675ec3d
Create Date: 2026-11-26

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a7c4e91b2d35'
down_revision: Union[str, Sequence[str], None] = '840b2675ec3d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # The server default fills the rows that already exist.
    op.add_column('chat_messages', sa.Column('status', sa.String(length=16), server_default='ok', nullable=False))
    op.add_column('chat_messages', sa.Column('error_code', sa.String(length=32), nullable=True))
    op.add_column('chat_messages', sa.Column('citations', sa.JSON(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('chat_messages', 'citations')
    op.drop_column('chat_messages', 'error_code')
    op.drop_column('chat_messages', 'status')
