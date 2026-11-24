# responses.py
# Creates consistent JSON success, health, and error responses.

from __future__ import annotations

from typing import Any

from flask import Response, jsonify

from model.model_service import PredictionResult


DISCLAIMER = (
    "This is a dataset-trained text classification, not factual verification. "
    "Confidence is model probability, not proof that a claim is true or false."
)


def prediction_response(result: PredictionResult) -> tuple[Response, int]:
    return (
        jsonify(
            {
                "ok": True,
                "prediction": result.to_dict(),
                "disclaimer": DISCLAIMER,
            }
        ),
        200,
    )


def health_response(
    *,
    model_loaded: bool,
    model_available: bool,
) -> tuple[Response, int]:
    status_code = 200 if model_available else 503
    return (
        jsonify(
            {
                "ok": model_available,
                "status": "ready" if model_available else "unavailable",
                "model_loaded": model_loaded,
            }
        ),
        status_code,
    )


def error_response(
    message: str,
    *,
    code: str,
    status_code: int,
    field: str | None = None,
    details: dict[str, Any] | None = None,
) -> tuple[Response, int]:
    error: dict[str, Any] = {
        "code": code,
        "message": message,
    }

    if field is not None:
        error["field"] = field

    if details:
        error["details"] = details

    return jsonify({"ok": False, "error": error}), status_code
