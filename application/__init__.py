# __init__.py
# Creates the Flask application, model service, routes, and HTTP error contract.

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Final

from flask import Flask, current_app, request
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from config import DefaultConfig
from model.model_service import ModelService


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
TEMPLATE_DIRECTORY: Final[Path] = PROJECT_ROOT / "templates"
STATIC_DIRECTORY: Final[Path] = PROJECT_ROOT / "static"
MODEL_SERVICE_EXTENSION_KEY: Final[str] = "model_service"

ModelServiceFactory = Callable[..., ModelService]


def create_app(
    config_overrides: Mapping[str, Any] | None = None,
    model_service_factory: ModelServiceFactory = ModelService,
) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(TEMPLATE_DIRECTORY),
        static_folder=str(STATIC_DIRECTORY),
        static_url_path="/static",
    )
    app.config.from_object(DefaultConfig)

    if config_overrides:
        app.config.from_mapping(config_overrides)

    _validate_configuration(app)

    service = model_service_factory(
        pipeline_path=Path(app.config["MODEL_PIPELINE_PATH"]),
        metadata_path=Path(app.config["MODEL_METADATA_PATH"]),
        maximum_characters=int(app.config["MODEL_MAX_INPUT_CHARACTERS"]),
    )
    app.extensions[MODEL_SERVICE_EXTENSION_KEY] = service

    if app.config["MODEL_EAGER_LOAD"]:
        service.load()

    from application.routes import main_blueprint

    app.register_blueprint(main_blueprint)
    _register_error_handlers(app)
    return app


def get_model_service() -> ModelService:
    service = current_app.extensions.get(MODEL_SERVICE_EXTENSION_KEY)

    if not isinstance(service, ModelService):
        raise RuntimeError("The application model service is not configured.")

    return service


def _register_error_handlers(app: Flask) -> None:
    from application.responses import error_response

    @app.errorhandler(RequestEntityTooLarge)
    def handle_request_too_large(_error: RequestEntityTooLarge):
        return error_response(
            "The request body is too large.",
            code="request_too_large",
            status_code=413,
        )

    @app.errorhandler(HTTPException)
    def handle_http_error(error: HTTPException):
        if request.path.startswith("/predict") or request.path.startswith("/health"):
            return error_response(
                error.description,
                code=f"http_{error.code}",
                status_code=int(error.code or 500),
            )

        return error


def _validate_configuration(app: Flask) -> None:
    required = (
        "MODEL_PIPELINE_PATH",
        "MODEL_METADATA_PATH",
        "MODEL_MAX_INPUT_CHARACTERS",
        "MODEL_EAGER_LOAD",
    )
    missing = [key for key in required if key not in app.config]

    if missing:
        raise RuntimeError(
            "Required application configuration is missing: "
            + ", ".join(sorted(missing))
        )

    maximum_characters = app.config["MODEL_MAX_INPUT_CHARACTERS"]

    if not isinstance(maximum_characters, int) or maximum_characters <= 0:
        raise RuntimeError(
            "MODEL_MAX_INPUT_CHARACTERS must be a positive integer."
        )

    maximum_content_length = app.config.get("MAX_CONTENT_LENGTH")

    if maximum_content_length is not None and (
        not isinstance(maximum_content_length, int)
        or maximum_content_length <= 0
    ):
        raise RuntimeError("MAX_CONTENT_LENGTH must be a positive integer.")
