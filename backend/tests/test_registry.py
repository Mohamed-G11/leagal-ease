"""Model Registry tests: promotion rule (pure) + register/load/predict/promote on a local MLflow."""
import mlflow
import pytest
from mlflow.tracking import MlflowClient

from app.core.schemas import Answer
from app.registry import promote as pr
from app.registry.promote import decide
from app.registry.pyfunc_model import LegalEaseRAGModel, _to_questions


def test_decide_rules():
    assert decide({"recall_at_5": 0.8}, None).promote  # first version
    base = {"recall_at_5": 0.80, "mrr": 0.60}
    assert decide({"recall_at_5": 0.93, "mrr": 0.73}, base).promote  # better on both
    assert not decide({"recall_at_5": 0.80, "mrr": 0.60}, base).promote  # equal: not better
    assert decide({"recall_at_5": 0.79, "mrr": 0.70}, base).promote  # tiny loss tolerated, mrr better
    d = decide({"recall_at_5": 0.70, "mrr": 0.90}, base)
    assert not d.promote and "recall_at_5 regressed" in d.reasons[0]
    # generation metric only compared when both runs have it
    assert not decide({"recall_at_5": 0.9, "mrr": 0.7, "abstention_correct_rate": 0.5},
                      {**base, "abstention_correct_rate": 0.9}).promote


def test_to_questions_accepts_common_inputs():
    pd = pytest.importorskip("pandas")
    assert _to_questions("q") == ["q"]
    assert _to_questions({"question": ["a", "b"]}) == ["a", "b"]
    assert _to_questions(pd.DataFrame({"question": ["x"]})) == ["x"]


class FakePipeline:
    def answer(self, question, filters=None):
        return Answer(answer=f"echo: {question}", status="answered", lang="fr")


def _log_version(tmp_path, metrics):
    with mlflow.start_run():
        mlflow.log_metrics(metrics)
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("name: test\n", encoding="utf-8")
        info = mlflow.pyfunc.log_model(name="rag_pipeline", python_model=LegalEaseRAGModel(FakePipeline()),
                                       artifacts={"config": str(cfg)}, registered_model_name=pr.MODEL_NAME,
                                       pip_requirements=["mlflow"])
    return str(info.registered_model_version)


def test_register_load_predict_and_promote(tmp_path, monkeypatch):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    mlflow.set_experiment("registry-test")  # do not reuse an experiment id cached by another test
    client = MlflowClient()

    v1 = _log_version(tmp_path, {"recall_at_5": 0.79, "mrr": 0.53})
    model = mlflow.pyfunc.load_model(f"models:/{pr.MODEL_NAME}/{v1}")
    out = model.predict({"question": ["Quel délai ?"]})
    assert out[0]["answer"] == "echo: Quel délai ?" and out[0]["status"] == "answered"

    assert pr.promote(v1, client).promote  # first version -> champion
    assert not pr.promote(v1, client).promote  # already champion
    assert str(client.get_model_version_by_alias(pr.MODEL_NAME, "champion").version) == v1

    v2 = _log_version(tmp_path, {"recall_at_5": 0.93, "mrr": 0.73})
    assert pr.promote(v2, client).promote
    assert str(client.get_model_version_by_alias(pr.MODEL_NAME, "champion").version) == v2
    assert client.get_model_version(pr.MODEL_NAME, v1).tags["role"] == "previous_champion"

    v3 = _log_version(tmp_path, {"recall_at_5": 0.70, "mrr": 0.90})  # regresses on recall
    assert not pr.promote(v3, client).promote
    assert str(client.get_model_version_by_alias(pr.MODEL_NAME, "challenger").version) == v3
    assert client.get_model_version(pr.MODEL_NAME, v3).tags["promotion_decision"] == "rejected"


def test_model_code_excludes_legacy_and_heavy_files(tmp_path):
    from app.registry.register import build_code_dir

    pkg = build_code_dir(tmp_path)
    files = {p.relative_to(pkg).as_posix() for p in pkg.rglob("*") if p.is_file()}
    assert "pipeline.py" in files and "core/schemas.py" in files and "retrieval/hybrid.py" in files
    assert not any(f.startswith(("rag/", "v1/", "api/")) for f in files)
    assert "core/config.py" not in files and not any(f.endswith((".pdf", ".db", ".pyc")) for f in files)
