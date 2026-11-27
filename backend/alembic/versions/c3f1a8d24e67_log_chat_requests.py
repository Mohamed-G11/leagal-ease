"""log chat requests in ai_query_logs and link feedback to messages

Revision ID: c3f1a8d24e67
Revises: a7c4e91b2d35
Create Date: 2026-11-27

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c3f1a8d24e67'
down_revision: Union[str, Sequence[str], None] = 'a7c4e91b2d35'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # All nullable: rows written before this migration stay valid.
    op.add_column('ai_query_logs', sa.Column('status', sa.String(length=16), nullable=True))
    op.add_column('ai_query_logs', sa.Column('error_code', sa.String(length=32), nullable=True))
    op.add_column('ai_query_logs', sa.Column('citations_count', sa.Integer(), nullable=True))
    op.add_column('ai_query_logs', sa.Column('latency_ms', sa.Integer(), nullable=True))
    op.add_column('ai_query_logs', sa.Column('reranker_score', sa.Float(), nullable=True))
    for table in ('ai_query_logs', 'ai_feedback'):
        op.add_column(table, sa.Column('message_id', sa.Integer(), nullable=True))
        op.create_index(op.f(f'ix_{table}_message_id'), table, ['message_id'], unique=False)
        # Deleting a conversation keeps the log and the rating, without the link.
        op.create_foreign_key(f'fk_{table}_message_id', table, 'chat_messages', ['message_id'], ['id'],
                              ondelete='SET NULL')


def downgrade() -> None:
    """Downgrade schema."""
    for table in ('ai_feedback', 'ai_query_logs'):
        op.drop_constraint(f'fk_{table}_message_id', table, type_='foreignkey')
        op.drop_index(op.f(f'ix_{table}_message_id'), table_name=table)
        op.drop_column(table, 'message_id')
    for column in ('reranker_score', 'latency_ms', 'citations_count', 'error_code', 'status'):
        op.drop_column('ai_query_logs', column)
