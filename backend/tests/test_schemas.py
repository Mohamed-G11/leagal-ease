"""Validation of the shared Answer contract (app/core/schemas.py)."""
import pytest
from pydantic import ValidationError

from app.core.schemas import Answer, Citation

CITATION = {"chunk_id": "loi_p2_0", "doc_title": "Loi 2018-52", "page": 2, "quote": "dans un delai de dix jours"}


def answer_data(**changes):
    return {"answer": "Dix jours [1].", "citations": [CITATION], "status": "answered", "lang": "fr", **changes}


def error_locations(data):
    with pytest.raises(ValidationError) as exc:
        Answer.model_validate(data)
    return [error["loc"] for error in exc.value.errors()]


def test_valid_answer():
    answer = Answer.model_validate(answer_data(confidence=0.8))
    assert answer.status == "answered" and answer.confidence == 0.8
    assert answer.citations == [Citation(**CITATION)]
    assert Answer.model_validate_json(answer.model_dump_json()) == answer


def test_abstained_answer_needs_no_citations():
    answer = Answer(answer="Je ne trouve pas.", status="abstained", lang="fr")
    assert answer.citations == [] and answer.confidence is None


@pytest.mark.parametrize("lang", ["fr", "ar", "en", "darija"])
def test_supported_languages(lang):
    assert Answer.model_validate(answer_data(lang=lang)).lang == lang


@pytest.mark.parametrize("field,value", [("status", "refused"), ("status", None), ("lang", "es")])
def test_invalid_status_or_language(field, value):
    assert error_locations(answer_data(**{field: value})) == [(field,)]


@pytest.mark.parametrize("field", ["answer", "status", "lang"])
def test_required_answer_fields(field):
    data = answer_data()
    del data[field]
    assert error_locations(data) == [(field,)]


@pytest.mark.parametrize("field", ["chunk_id", "doc_title", "page", "quote"])
def test_required_citation_fields(field):
    citation = {key: value for key, value in CITATION.items() if key != field}
    assert error_locations(answer_data(citations=[citation])) == [("citations", 0, field)]


@pytest.mark.parametrize("field,value", [("page", "two"), ("page", None), ("quote", None), ("chunk_id", 12)])
def test_mistyped_citation_fields(field, value):
    assert error_locations(answer_data(citations=[{**CITATION, field: value}])) == [("citations", 0, field)]
