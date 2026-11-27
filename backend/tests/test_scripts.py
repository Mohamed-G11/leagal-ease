"""The command-line scripts must be able to load the models. Each runs in a fresh process, against SQLite."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]


def run(tmp_path, *args):
    env = {**os.environ, "DATABASE_URL": f"sqlite:///{(tmp_path / 'scripts.db').as_posix()}", "SECRET_KEY": "test",
           "PYTHONPATH": str(BACKEND), "PYTHONUTF8": "1"}
    return subprocess.run([sys.executable, *args], cwd=BACKEND, env=env, capture_output=True, text=True,
                          encoding="utf-8")


@pytest.mark.parametrize("module", ["scripts.make_admin", "scripts.seed_procedure"])
def test_script_models_resolve(tmp_path, module):
    code = f"import {module}; from sqlalchemy.orm import configure_mappers; configure_mappers()"
    done = run(tmp_path, "-c", code)
    assert done.returncode == 0, done.stderr[-500:]


def test_seed_then_make_admin(tmp_path):
    seeded = run(tmp_path, "-m", "scripts.seed_procedure")
    assert seeded.returncode == 0 and "Seeding complete" in seeded.stdout, seeded.stderr[-500:]
    missing = run(tmp_path, "-m", "scripts.make_admin", "nobody")
    assert missing.returncode == 1 and "not found" in missing.stdout, missing.stderr[-500:]
