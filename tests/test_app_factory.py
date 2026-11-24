# test_app_factory.py
# Verifies Flask factory configuration and model-service lifecycle ownership.

from __future__ import annotations

from pathlib import Path

import pytest
from flask import Flask

from application import (
    MODEL_SERVICE_EXTENSION_KEY,
    create_app,
    get_model_service,
)
from model.model_service import ModelService


class FakeModelService(ModelService):
    """Test double that records factory parameters without loading Joblib."""

    def __init__(
        self,
        pipeline_path: Path,
        metadata_path: Path,
        maximum_characters: int,
    ) -> None:
        self.pipeline_path = Path(pipeline_path)
        self.metadata_path = Path(metadata_path)
        self.maximum_characters = maximum_characters
        self.load_calls = 0

    @property
    def is_loaded(self) -> bool:
        return self.load_calls > 0

    def load(self) -> None:
        self.load_calls += 1


def make_app(**overrides: object) -> Flask:
    config = {
        "TESTING": True,
        "MODEL_PIPELINE_PATH": Path("test-pipeline.joblib"),
        "MODEL_METADATA_PATH": Path("test-metadata.json"),
        "MODEL_MAX_INPUT_CHARACTERS": 1_234,
        "MODEL_EAGER_LOAD": False,
    }
    config.update(overrides)
    return create_app(config, model_service_factory=FakeModelService)


def test_create_app_returns_flask_application() -> None:
    app = make_app()
    assert isinstance(app, Flask)
    assert app.testing is True


def test_factory_uses_physical_template_and_static_directories() -> None:
    app = make_app()
    assert Path(app.template_folder).name == "templates"
    assert Path(app.static_folder).name == "static"
    assert app.static_url_path == "/static"


def test_factory_applies_configuration_overrides() -> None:
    app = make_app(MODEL_MAX_INPUT_CHARACTERS=9_999)
    assert app.config["MODEL_MAX_INPUT_CHARACTERS"] == 9_999


def test_factory_registers_model_service_extension() -> None:
    app = make_app()
    service = app.extensions[MODEL_SERVICE_EXTENSION_KEY]
    assert isinstance(service, FakeModelService)
    assert service.pipeline_path == Path("test-pipeline.joblib")
    assert service.metadata_path == Path("test-metadata.json")
    assert service.maximum_characters == 1_234


def test_factory_keeps_model_loading_lazy_by_default() -> None:
    app = make_app()
    service = app.extensions[MODEL_SERVICE_EXTENSION_KEY]
    assert service.load_calls == 0


def test_factory_can_eagerly_validate_model() -> None:
    app = make_app(MODEL_EAGER_LOAD=True)
    service = app.extensions[MODEL_SERVICE_EXTENSION_KEY]
    assert service.load_calls == 1


def test_get_model_service_returns_application_scoped_service() -> None:
    app = make_app()
    expected = app.extensions[MODEL_SERVICE_EXTENSION_KEY]

    with app.app_context():
        assert get_model_service() is expected


def test_get_model_service_requires_application_context() -> None:
    with pytest.raises(RuntimeError):
        get_model_service()


def test_factory_rejects_non_positive_model_input_limit() -> None:
    with pytest.raises(RuntimeError, match="positive integer"):
        make_app(MODEL_MAX_INPUT_CHARACTERS=0)


def test_factory_rejects_invalid_content_length() -> None:
    with pytest.raises(RuntimeError, match="MAX_CONTENT_LENGTH"):
        make_app(MAX_CONTENT_LENGTH=-1)


def test_factory_instances_do_not_share_service_state() -> None:
    first = make_app()
    second = make_app()
    assert (
        first.extensions[MODEL_SERVICE_EXTENSION_KEY]
        is not second.extensions[MODEL_SERVICE_EXTENSION_KEY]
    )
