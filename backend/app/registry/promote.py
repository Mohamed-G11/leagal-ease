"""Promotion rule for the Model Registry (W4.4).

Every new version is a candidate. It becomes `champion` only if it is at least as good
as the current champion on every gate metric (with a small tolerance) and strictly
better on at least one. Otherwise it gets the `challenger` alias. The decision and its
reasons are written on the version as tags, so the history is auditable.

    python -m app.registry.promote --version 3
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field

from mlflow.tracking import MlflowClient

MODEL_NAME = "legalease-rag"
CHAMPION, CHALLENGER = "champion", "challenger"

# metric -> tolerance (a candidate may lose up to this much without being rejected)
GATE = {
    "recall_at_5": 0.02,
    "mrr": 0.02,
    "abstention_correct_rate": 0.05,  # only compared when both runs have generation metrics
    "citation_hit_rate": 0.05,
}
PRIMARY = ("recall_at_5", "mrr")


@dataclass
class Decision:
    promote: bool
    reasons: list[str] = field(default_factory=list)


def decide(candidate: dict[str, float], champion: dict[str, float] | None) -> Decision:
    """Pure function: easy to unit-test and to explain during the validation."""
    if not champion:
        return Decision(True, ["no champion yet"])
    reasons, better = [], False
    for metric, tol in GATE.items():
        if metric not in candidate or metric not in champion:
            continue
        c, ch = candidate[metric], champion[metric]
        if c < ch - tol:
            return Decision(False, [f"{metric} regressed: {c:.3f} < {ch:.3f} - {tol}"])
        if c > ch and metric in PRIMARY:
            better = True
        reasons.append(f"{metric}: {c:.3f} vs {ch:.3f}")
    if not better:
        return Decision(False, reasons + ["not better than champion on recall_at_5 or mrr"])
    return Decision(True, reasons)


def _run_metrics(client: MlflowClient, version) -> dict[str, float]:
    return dict(client.get_run(version.run_id).data.metrics) if version.run_id else {}


def _set_role(client: MlflowClient, version: str, alias: str) -> None:
    """Aliases need a recent MLflow server; fall back to a tag if the server refuses them."""
    try:
        client.set_registered_model_alias(MODEL_NAME, alias, version)
    except Exception as exc:  # noqa: BLE001
        print(f"alias '{alias}' not supported by the server ({exc.__class__.__name__}); using a tag")
        client.set_model_version_tag(MODEL_NAME, version, "role", alias)


def current_champion(client: MlflowClient):
    try:
        return client.get_model_version_by_alias(MODEL_NAME, CHAMPION)
    except Exception:  # noqa: BLE001
        tagged = [v for v in client.search_model_versions(f"name='{MODEL_NAME}'")
                  if v.tags.get("role") == CHAMPION]
        return max(tagged, key=lambda v: int(v.version)) if tagged else None


def promote(version: str, client: MlflowClient | None = None) -> Decision:
    client = client or MlflowClient()
    candidate = client.get_model_version(MODEL_NAME, version)
    champion = current_champion(client)
    if champion is not None and str(champion.version) == str(version):
        return Decision(False, ["already champion"])

    decision = decide(_run_metrics(client, candidate), _run_metrics(client, champion) if champion else None)
    if decision.promote:
        if champion is not None:
            client.set_model_version_tag(MODEL_NAME, str(champion.version), "role", "previous_champion")
        _set_role(client, str(version), CHAMPION)
    else:
        _set_role(client, str(version), CHALLENGER)

    client.set_model_version_tag(MODEL_NAME, str(version), "promotion_decision",
                                 "promoted" if decision.promote else "rejected")
    client.set_model_version_tag(MODEL_NAME, str(version), "promotion_reasons", " | ".join(decision.reasons)[:4000])
    print(("PROMOTED" if decision.promote else "KEPT AS CHALLENGER") + f" v{version}: " + "; ".join(decision.reasons))
    return decision


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--version", required=True)
    promote(p.parse_args().version)
