"""create step_documents and procedure_collaborators

The older migrations 37c563bd88fe and 840b2675ec3d are empty, so these tables were never created by Alembic.
scripts/seed_procedure.py calls Base.metadata.create_all, so some databases already have them: only the
missing tables are created.

Revision ID: e5b9c2d71f40
Revises: c3f1a8d24e67
Create Date: 2026-11-27

"""
from typing import Sequence, Union

from alembic import context, op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e5b9c2d71f40'
down_revision: Union[str, Sequence[str], None] = 'c3f1a8d24e67'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _exists(table: str) -> bool:
    # Offline mode (--sql) has no database to look at: emit the full SQL.
    return not context.is_offline_mode() and sa.inspect(op.get_bind()).has_table(table)


def upgrade() -> None:
    """Upgrade schema."""
    if not _exists('step_documents'):
        op.create_table(
            'step_documents',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('step_progress_id', sa.Integer(), nullable=False),
            sa.Column('original_filename', sa.String(), nullable=False),
            sa.Column('stored_filename', sa.String(), nullable=False),
            sa.Column('content_type', sa.String(), nullable=False),
            sa.Column('file_size', sa.BigInteger(), nullable=False),
            sa.Column('uploaded_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
            sa.ForeignKeyConstraint(['step_progress_id'], ['user_step_progress.id']),
            sa.PrimaryKeyConstraint('id'),
            sa.UniqueConstraint('stored_filename'),
        )
        op.create_index(op.f('ix_step_documents_id'), 'step_documents', ['id'], unique=False)

    if not _exists('procedure_collaborators'):
        op.create_table(
            'procedure_collaborators',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('procedure_id', sa.Integer(), nullable=False),
            sa.Column('user_id', sa.Integer(), nullable=False),
            sa.Column('role', sa.String(length=20), nullable=False),
            sa.Column('invited_by', sa.Integer(), nullable=True),
            sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
            sa.ForeignKeyConstraint(['invited_by'], ['users.id']),
            sa.ForeignKeyConstraint(['procedure_id'], ['user_procedures.id'], ondelete='CASCADE'),
            sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
            sa.PrimaryKeyConstraint('id'),
            sa.UniqueConstraint('procedure_id', 'user_id', name='uq_procedure_collaborator'),
        )
        for column in ('id', 'procedure_id', 'user_id'):
            op.create_index(op.f(f'ix_procedure_collaborators_{column}'), 'procedure_collaborators', [column],
                            unique=False)


def downgrade() -> None:
    """Downgrade schema. Refuses to drop a table that holds data."""
    tables = ('procedure_collaborators', 'step_documents')
    if not context.is_offline_mode():
        # Check both before dropping either.
        for table in tables:
            if _exists(table) and op.get_bind().execute(sa.text(f'SELECT 1 FROM {table} LIMIT 1')).first():
                raise RuntimeError(f'{table} is not empty: back it up and empty it before downgrading.')
    for table in tables:
        op.drop_table(table)
