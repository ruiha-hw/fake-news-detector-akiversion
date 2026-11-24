# test_http_routes.py
# Verifies HTML routes, health status, prediction requests, and JSON errors.

from __future__ import annotations

from pathlib import Path

import pytest

from application import create_app
from model.model_service import (
    ModelIntegrityError,
    ModelService,
    PredictionResult,
)


class RouteModelService(ModelService):
    def __init__(
        self,
        pipeline_path: Path,
        metadata_path: Path,
        maximum_characters: int,
    ) -> None:
        self.maximum_characters = maximum_characters
        self._loaded = False
        self.fail_load = False
        self.fail_predict = False
        self.last_input: tuple[str, str] | None = None

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        if self.fail_load:
            raise ModelIntegrityError("test integrity failure")
        self._loaded = True

    def predict(self, title: object = "", article: object = "") -> PredictionResult:
        if self.fail_predict:
            raise ModelIntegrityError("test prediction failure")
        self.load()
        self.last_input = (str(title), str(article))
        return PredictionResult(
            label=1,
            classification="likely real",
            confidence=0.75,
            fake_probability=0.25,
            real_probability=0.75,
            input_characters=len(str(title)) + len(str(article)),
        )


def make_app(**overrides: object):
    config = {
        "TESTING": True,
        "MODEL_PIPELINE_PATH": Path("pipeline.joblib"),
        "MODEL_METADATA_PATH": Path("metadata.json"),
        "MODEL_MAX_INPUT_CHARACTERS": 100,
        "MODEL_EAGER_LOAD": False,
        "MAX_CONTENT_LENGTH": 1024,
    }
    config.update(overrides)
    return create_app(config, model_service_factory=RouteModelService)


def test_route_map_contains_public_contract() -> None:
    app = make_app()
    rules = {str(rule): sorted(rule.methods) for rule in app.url_map.iter_rules()}
    assert "/" in rules
    assert "/about" in rules
    assert "/health" in rules
    assert "/predict" in rules
    assert "POST" in rules["/predict"]


def test_home_and_about_render_successfully() -> None:
    client = make_app().test_client()
    assert client.get("/").status_code == 200
    assert client.get("/about").status_code == 200


def test_health_loads_model_and_returns_ready() -> None:
    app = make_app()
    response = app.test_client().get("/health")
    assert response.status_code == 200
    assert response.get_json() == {
        "ok": True,
        "status": "ready",
        "model_loaded": True,
    }


def test_health_returns_503_when_model_is_unavailable() -> None:
    app = make_app()
    app.extensions["model_service"].fail_load = True
    response = app.test_client().get("/health")
    assert response.status_code == 503
    assert response.get_json()["status"] == "unavailable"


def test_predict_accepts_json() -> None:
    app = make_app()
    response = app.test_client().post(
        "/predict",
        json={"title": "Headline", "article": "Article body"},
    )
    payload = response.get_json()
    assert response.status_code == 200
    assert payload["ok"] is True
    assert payload["prediction"]["classification"] == "likely real"
    assert "not factual verification" in payload["disclaimer"]
    assert app.extensions["model_service"].last_input == (
        "Headline",
        "Article body",
    )


def test_predict_accepts_form_data() -> None:
    response = make_app().test_client().post(
        "/predict",
        data={"title": "Headline", "article": "Article body"},
    )
    assert response.status_code == 200
    assert response.get_json()["prediction"]["label"] == 1


def test_predict_accepts_text_alias_for_json_article() -> None:
    response = make_app().test_client().post(
        "/predict",
        json={"title": "Headline", "text": "Article body"},
    )
    assert response.status_code == 200


@pytest.mark.parametrize(
    ("payload", "field"),
    [
        ({"title": 123, "article": "body"}, "title"),
        ({"title": "headline", "article": 123}, "article"),
    ],
)
def test_predict_rejects_non_string_fields(payload: dict, field: str) -> None:
    response = make_app().test_client().post("/predict", json=payload)
    assert response.status_code == 400
    assert response.get_json()["error"]["field"] == field


def test_predict_rejects_empty_input() -> None:
    response = make_app().test_client().post(
        "/predict",
        json={"title": " ", "article": "\n"},
    )
    assert response.status_code == 422
    assert response.get_json()["error"]["code"] == "empty_input"


def test_predict_rejects_excessive_input() -> None:
    response = make_app().test_client().post(
        "/predict",
        json={"title": "x" * 101, "article": ""},
    )
    assert response.status_code == 422
    assert response.get_json()["error"]["code"] == "input_too_long"


def test_predict_rejects_unsupported_content_type() -> None:
    response = make_app().test_client().post(
        "/predict",
        data="plain text",
        content_type="text/plain",
    )
    assert response.status_code == 415
    assert response.get_json()["error"]["code"] == "unsupported_content_type"


def test_predict_rejects_invalid_json() -> None:
    response = make_app().test_client().post(
        "/predict",
        data="{broken",
        content_type="application/json",
    )
    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "invalid_json"


def test_predict_maps_model_integrity_failure_to_503() -> None:
    app = make_app()
    app.extensions["model_service"].fail_predict = True
    response = app.test_client().post(
        "/predict",
        json={"title": "Headline", "article": "Body"},
    )
    assert response.status_code == 503
    assert response.get_json()["error"]["code"] == "model_unavailable"


def test_predict_get_returns_json_405() -> None:
    response = make_app().test_client().get("/predict")
    assert response.status_code == 405
    assert response.is_json
    assert response.get_json()["error"]["code"] == "http_405"


def test_oversized_request_returns_json_413() -> None:
    app = make_app(MAX_CONTENT_LENGTH=32)
    response = app.test_client().post(
        "/predict",
        data="x" * 64,
        content_type="application/json",
    )
    assert response.status_code == 413
    assert response.get_json()["error"]["code"] == "request_too_large"
