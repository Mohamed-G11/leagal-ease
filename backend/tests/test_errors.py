"""Error format tests. They use a throwaway app, so no database is needed."""
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient
from slowapi.errors import RateLimitExceeded
from sqlalchemy.exc import OperationalError

from app.core.errors import APIError, register_error_handlers


@pytest.fixture(scope="module")
def client():
    app = FastAPI()
    register_error_handlers(app)
    app.add_middleware(CORSMiddleware, allow_origins=["*"])

    @app.get("/items/{item_id}")
    def item(item_id: int):
        raise HTTPException(401, "Nope", headers={"WWW-Authenticate": "Bearer"})

    @app.get("/coded")
    def coded():
        raise APIError(503, "rag_loading", "Still loading.")

    @app.get("/status/{status}")
    def with_status(status: int):
        raise HTTPException(status, "msg")

    @app.get("/limited")
    def limited():
        raise RateLimitExceeded(SimpleNamespace(error_message=None, limit="5 per 1 minute"))

    @app.get("/db")
    def db():
        raise OperationalError("SELECT secret", {}, Exception("password in here"))

    @app.get("/crash")
    def crash():
        raise ValueError("internal secret")

    return TestClient(app)


def test_http_error_keeps_detail_and_headers(client):
    resp = client.get("/items/1")
    assert resp.status_code == 401
    assert resp.json() == {"error": {"code": "unauthorized", "message": "Nope"}, "detail": "Nope"}
    assert resp.headers["www-authenticate"] == "Bearer"


def test_unknown_route_and_explicit_code(client):
    assert client.get("/nowhere").json()["error"]["code"] == "not_found"
    resp = client.get("/coded")
    assert resp.status_code == 503 and resp.json()["error"]["code"] == "rag_loading"


@pytest.mark.parametrize("status,code", [(409, "conflict"), (429, "rate_limited"), (400, "bad_request"),
                                         (405, "bad_request")])
def test_codes_by_status(client, status, code):
    assert client.get(f"/status/{status}").json()["error"]["code"] == code


def test_slowapi_limit_uses_the_same_shape(client):
    resp = client.get("/limited")
    assert resp.status_code == 429
    assert resp.json()["error"] == {"code": "rate_limited", "message": "5 per 1 minute"}


def test_validation_error(client):
    body = client.get("/items/abc").json()
    assert body["error"]["code"] == "validation_error"
    assert body["detail"] == body["error"]["details"]
    assert body["detail"][0]["loc"] == ["path", "item_id"]


@pytest.mark.parametrize("path,status,code", [("/db", 503, "database_unavailable"), ("/crash", 500, "internal_error")])
def test_server_errors_do_not_leak(client, path, status, code):
    resp = client.get(path, headers={"Origin": "http://frontend.test"})
    assert resp.status_code == status and resp.json()["error"]["code"] == code
    assert "secret" not in resp.text and "password" not in resp.text
    assert resp.headers["access-control-allow-origin"] == "*"
