from fastapi.testclient import TestClient

from rag2.main import create_app


def test_health_endpoint() -> None:
    client = TestClient(create_app())
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_page_renders() -> None:
    client = TestClient(create_app())
    response = client.get("/")
    assert response.status_code == 200
    assert "Logistics Document QA" in response.text
    assert "Document Chat" in response.text
    assert "Context and guardrails" in response.text
