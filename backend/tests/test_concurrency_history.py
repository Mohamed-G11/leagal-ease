"""Thread safety of the shared pipeline + conversation history (offline fakes)."""
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from app.core.experiment_config import ExperimentConfig
from app.core.schemas import ChatTurn, Chunk, ScoredChunk, recent_history
from app.generation.generator import Generator
from app.pipeline import RAGPipeline
from app.retrieval.advanced import AdvancedRetriever


class StaticRetriever:
    client = None

    def retrieve(self, query, k=5, filters=None):
        return [ScoredChunk(chunk=Chunk(chunk_id="d_p1_0", doc_id="d", doc_title="D", page=1, lang="fr",
                                        text="Le délai est de dix jours."), score=0.9)]


class SlowTransformer:
    """Language depends on the question; the sleep makes the two requests overlap."""

    def __init__(self):
        self.seen_history = []

    def transform(self, question, history=None):
        self.seen_history.append(history)
        lang = "darija" if "9adech" in question else "fr"
        time.sleep(0.2 if lang == "darija" else 0.05)
        return [question], lang


class EchoLLM:
    """Records the messages; answers in whatever language the system prompt asked for."""

    def __init__(self):
        self.calls = []
        self.lock = threading.Lock()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        with self.lock:
            self.calls.append(kwargs["messages"])
        payload = {"status": "answered", "answer": "Dix jours [1].", "citations": [{"source": 1, "quote": "dix jours"}]}
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))])


def make_pipeline(transformer, llm):
    cfg = ExperimentConfig()
    cfg.query_transform.enabled = True
    retriever = AdvancedRetriever(cfg, StaticRetriever(), transformer=transformer, reranker=None)
    return RAGPipeline(cfg, retriever=retriever, generator=Generator(cfg.llm, client=llm))


def test_concurrent_requests_keep_their_own_language():
    pipe = make_pipeline(SlowTransformer(), EchoLLM())
    questions = ["9adech nestanna ?", "Quel est le délai ?"] * 5
    with ThreadPoolExecutor(max_workers=10) as pool:
        answers = list(pool.map(pipe.answer, questions))
    for q, a in zip(questions, answers, strict=True):
        assert a.lang == ("darija" if "9adech" in q else "fr"), q
    assert not hasattr(pipe.retriever, "last_lang")


def test_history_reaches_rewriter_and_llm_and_is_bounded():
    transformer, llm = SlowTransformer(), EchoLLM()
    pipe = make_pipeline(transformer, llm)
    history = [{"role": "user", "content": f"question {i}"} if i % 2 == 0
               else {"role": "assistant", "content": "x" * 2000} for i in range(10)]
    pipe.answer("Et pour une SARL ?", history=history)

    sent_to_rewriter = transformer.seen_history[-1]
    assert len(sent_to_rewriter) == pipe.cfg.llm.max_history_messages
    messages = llm.calls[-1]
    assert messages[0]["role"] == "system" and "context only" in messages[0]["content"]
    assert [m["role"] for m in messages[1:-1]] == [t.role for t in sent_to_rewriter]
    assert all(len(m["content"]) <= 600 for m in messages[1:-1])  # long answers are cut
    assert messages[-1]["content"].startswith("Sources:")


def test_no_history_means_no_extra_messages():
    llm = EchoLLM()
    make_pipeline(SlowTransformer(), llm).answer("Quel est le délai ?")
    assert len(llm.calls[-1]) == 2 and "context only" not in llm.calls[-1][0]["content"]


def test_recent_history_helper():
    assert recent_history(None, 6) == []
    turns = recent_history([ChatTurn(role="user", content="a"), {"role": "assistant", "content": " "}], 6)
    assert [t.content for t in turns] == ["a"]  # blank messages dropped
