# routes.py
# Defines public HTML pages, health status, and the prediction endpoint.

from __future__ import annotations

from flask import Blueprint, current_app, render_template, request

from application.responses import (
    error_response,
    health_response,
    prediction_response,
)
from application.validation import (
    RequestValidationError,
    extract_prediction_input,
)
from model.model_service import ModelIntegrityError, ModelServiceError


main_blueprint = Blueprint("main", __name__)


@main_blueprint.get("/")
def home():
    return render_template(
        "home.html",
        maximum_characters=current_app.config["MODEL_MAX_INPUT_CHARACTERS"],
    )


@main_blueprint.get("/about")
def about():
    return render_template("about.html")


@main_blueprint.get("/health")
def health():
    service = _service()

    try:
        service.load()
    except ModelServiceError:
        current_app.logger.exception("Model health validation failed.")
        return health_response(
            model_loaded=False,
            model_available=False,
        )

    return health_response(
        model_loaded=service.is_loaded,
        model_available=True,
    )


@main_blueprint.post("/predict")
def predict():
    try:
        values = extract_prediction_input(
            request,
            int(current_app.config["MODEL_MAX_INPUT_CHARACTERS"]),
        )
        result = _service().predict(values.title, values.article)
        return prediction_response(result)
    except RequestValidationError as error:
        return error_response(
            error.message,
            code=error.code,
            status_code=error.status_code,
            field=error.field,
        )
    except ModelIntegrityError:
        current_app.logger.exception("Model integrity failure during prediction.")
        return error_response(
            "The classification model is unavailable.",
            code="model_unavailable",
            status_code=503,
        )
    except ModelServiceError:
        current_app.logger.exception("Model service failure during prediction.")
        return error_response(
            "The classification request could not be completed.",
            code="prediction_failed",
            status_code=500,
        )


def _service():
    from application import get_model_service

    return get_model_service()
