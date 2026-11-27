"""Shared test setup."""
import httpx
import pytest


@pytest.fixture(autouse=True)
def no_real_model_registry(monkeypatch):
    """A developer's .env may point at DagsHub: tests must not reach it."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_REGISTRY_URI", raising=False)


@pytest.fixture(autouse=True)
def no_real_groq_calls(monkeypatch):
    """Safety net: a test that forgets to mock Groq fails instead of calling the real API."""
    real_sync = httpx.HTTPTransport.handle_request
    real_async = httpx.AsyncHTTPTransport.handle_async_request

    def check(request):
        if request.url.host.endswith("groq.com"):
            pytest.fail(f"A test tried to call the real Groq API: {request.url}")

    def sync_guard(self, request):
        check(request)
        return real_sync(self, request)

    async def async_guard(self, request):
        check(request)
        return await real_async(self, request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", sync_guard)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", async_guard)
