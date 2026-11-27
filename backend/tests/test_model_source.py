"""Serving the registry champion: fallback, reload and the admin endpoint. Fakes only, no network."""
import threading
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.dependencies import get_current_user
from app.core.errors import APIError, register_error_handlers
from app.services import model_source
from app.services.rag import RagService
from app.v1.endpoints import admin


def fake_pipeline(name):
    cfg = SimpleNamespace(name=name, qdrant=SimpleNamespace(url="http://qdrant.test", collection=name))
    return SimpleNamespace(cfg=cfg)


@pytest.fixture
def registry(monkeypatch, tmp_path):
    """A fake registry: set `champion` to a version, None, or an exception."""
    state = SimpleNamespace(champion="1", loaded=[], broken=set())

    def champion_version():
        if isinstance(state.champion, Exception):
            raise state.champion
        return state.champion

    def load_version(version):
        if version in state.broken:
            raise RuntimeError("cannot load")
        state.loaded.append(version)
        return fake_pipeline(f"registry_v{version}")

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://registry.test")
    monkeypatch.setattr(model_source, "champion_version", champion_version)
    monkeypatch.setattr(model_source, "load_version", load_version)
    monkeypatch.setattr("app.pipeline.RAGPipeline", lambda cfg: fake_pipeline(cfg.name))
    config = tmp_path / "cfg.yaml"
    config.write_text("name: from_file\n", encoding="utf-8")
    state.service = RagService(str(config))
    return state


def served(service):
    return tuple(service.model_info().values())


def test_the_champion_is_served(registry):
    registry.service.load()
    assert served(registry.service) == ("registry", "1", "registry_v1")
    assert registry.service.state == "ready"


@pytest.mark.parametrize("champion", [None, ConnectionError("unreachable"), "broken"])
def test_falls_back_to_the_config_file(registry, champion):
    registry.champion = champion
    registry.broken.add("broken")
    registry.service.load()
    assert served(registry.service) == ("config", None, "from_file")
    assert registry.service.state == "ready"


def test_registry_is_off_without_a_tracking_uri(registry, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI")
    registry.champion = AssertionError("must not be asked")
    registry.service.load()
    assert served(registry.service) == ("config", None, "from_file")
    assert registry.service.reload() == "MLFLOW_TRACKING_URI is not set"


def test_reload_follows_the_alias(registry):
    registry.service.load()
    first = registry.service.pipeline
    assert registry.service.reload() is None and registry.service.pipeline is first  # same version: not rebuilt

    registry.champion = "2"
    assert registry.service.reload() is None
    assert served(registry.service) == ("registry", "2", "registry_v2")
    assert registry.loaded == ["1", "2"]


@pytest.mark.parametrize("champion,reason", [
    (None, "no champion"),
    (ConnectionError("unreachable"), "could not be loaded"),
    ("broken", "could not be loaded"),
])
def test_a_failed_reload_keeps_what_is_served(registry, champion, reason):
    registry.service.load()
    registry.champion = champion
    registry.broken.add("broken")
    assert reason in registry.service.reload()
    assert served(registry.service) == ("registry", "1", "registry_v1")


def test_reload_moves_from_the_config_file_to_the_registry(registry):
    registry.champion = None
    registry.service.load()
    registry.champion = "3"
    assert registry.service.reload() is None
    assert served(registry.service) == ("registry", "3", "registry_v3")


def test_only_one_load_at_a_time(registry, monkeypatch):
    registry.service.load()
    started, release = threading.Event(), threading.Event()

    def slow(version):
        started.set()
        release.wait(timeout=10)
        return fake_pipeline("slow")

    monkeypatch.setattr(model_source, "load_version", slow)
    registry.champion = "2"
    running = threading.Thread(target=registry.service.reload)
    running.start()
    assert started.wait(timeout=10)
    try:
        with pytest.raises(APIError) as exc:
            registry.service.reload()
        assert exc.value.status_code == 409
        assert registry.service.version == "1"  # still served while the next one loads
    finally:
        release.set()
        running.join(timeout=10)
    assert registry.service.version == "2"


@pytest.fixture
def client(registry, monkeypatch):
    monkeypatch.setattr(admin, "rag_service", registry.service)
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(admin.router, prefix="/api/v1")
    return app, TestClient(app)


URL = "/api/v1/admin/rag/reload"


def test_reload_endpoint_needs_an_admin(registry, client):
    app, http = client
    assert http.post(URL).status_code == 401
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(is_admin=False)
    assert http.post(URL).status_code == 403
    assert registry.loaded == []


def test_reload_endpoint_reports_the_switch(registry, client):
    app, http = client
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(is_admin=True)
    registry.service.load()
    registry.champion = "2"
    assert http.post(URL).json() == {
        "changed": True, "source": "registry", "version": "2", "config": "registry_v2", "message": None}
    assert http.post(URL).json()["changed"] is False


def test_champion_is_read_and_loaded_from_a_local_registry(tmp_path):
    """The real MLflow calls, against a SQLite store in a temporary folder (no server, no network)."""
    import mlflow
    from mlflow.tracking import MlflowClient

    from app.registry.promote import MODEL_NAME
    from app.registry.pyfunc_model import LegalEaseRAGModel

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    previous = mlflow.get_tracking_uri(), mlflow.get_registry_uri()
    try:
        mlflow.set_tracking_uri(uri)
        mlflow.set_registry_uri(uri)
        mlflow.set_experiment("model-source-test")
        assert model_source.champion_version() is None  # nothing registered
        with mlflow.start_run():
            mlflow.pyfunc.log_model(name="rag_pipeline", python_model=LegalEaseRAGModel(fake_pipeline("logged")),
                                    registered_model_name=MODEL_NAME, pip_requirements=["mlflow"])
        assert model_source.champion_version() is None  # registered, no alias yet
        MlflowClient().set_registered_model_alias(MODEL_NAME, "champion", "1")
        assert model_source.champion_version() == "1"
        assert model_source.load_version("1").cfg.name == "logged"
    finally:
        mlflow.set_tracking_uri(previous[0])
        mlflow.set_registry_uri(previous[1])
