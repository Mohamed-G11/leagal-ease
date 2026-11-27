"""The migration that creates step_documents and procedure_collaborators, run on in-memory SQLite."""
import importlib.util
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

PATH = Path(__file__).resolve().parents[1] / "alembic" / "versions" / \
    "e5b9c2d71f40_create_step_documents_and_collaborators.py"
spec = importlib.util.spec_from_file_location("missing_tables_migration", PATH)
migration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migration)

TABLES = {"step_documents", "procedure_collaborators"}


@pytest.fixture
def connection(monkeypatch):
    engine = sa.create_engine("sqlite://")
    with engine.connect() as conn:
        for parent in ("users", "user_procedures", "user_step_progress"):
            conn.execute(sa.text(f"CREATE TABLE {parent} (id INTEGER PRIMARY KEY)"))
        # No Alembic environment is running here, so "offline mode" has to be answered directly.
        monkeypatch.setattr(migration.context, "is_offline_mode", lambda: False)
        with Operations.context(MigrationContext.configure(conn)):
            yield conn


def tables(conn):
    return TABLES & set(sa.inspect(conn).get_table_names())


def test_upgrade_and_downgrade(connection):
    migration.upgrade()
    assert tables(connection) == TABLES
    migration.downgrade()
    assert tables(connection) == set()


def test_upgrade_accepts_tables_that_already_exist(connection):
    connection.execute(sa.text("CREATE TABLE step_documents (id INTEGER PRIMARY KEY)"))
    migration.upgrade()
    assert tables(connection) == TABLES


def test_downgrade_keeps_both_tables_when_one_has_rows(connection):
    migration.upgrade()
    connection.execute(sa.text(
        "INSERT INTO step_documents (step_progress_id, original_filename, stored_filename, content_type, file_size) "
        "VALUES (1, 'a.pdf', 'x.pdf', 'application/pdf', 10)"
    ))
    with pytest.raises(RuntimeError, match="step_documents is not empty"):
        migration.downgrade()
    assert tables(connection) == TABLES
