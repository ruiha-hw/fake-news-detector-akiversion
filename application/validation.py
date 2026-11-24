# validation.py
# Extracts and validates prediction input from JSON or HTML form requests.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from flask import Request

from model.model_service import MAX_INPUT_CHARACTERS, ModelInputError


@dataclass(frozen=True)
class PredictionInput:
    """Validated request values passed to the runtime model service."""

    title: str
    article: str


class RequestValidationError(ValueError):
    """Represents a client-correctable HTTP request error."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "invalid_request",
        field: str | None = None,
        status_code: int = 400,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.field = field
        self.status_code = status_code


def extract_prediction_input(
    request: Request,
    maximum_characters: int = MAX_INPUT_CHARACTERS,
) -> PredictionInput:
    """Reads JSON or form data and enforces the public prediction contract."""

    payload = _extract_payload(request)
    title = payload.get("title", "")
    article = payload.get("article", payload.get("text", ""))

    if not isinstance(title, str):
        raise RequestValidationError(
            "The title field must be a string.",
            code="invalid_field_type",
            field="title",
        )

    if not isinstance(article, str):
        raise RequestValidationError(
            "The article field must be a string.",
            code="invalid_field_type",
            field="article",
        )

    try:
        from model.model_service import prepare_runtime_input

        prepare_runtime_input(title, article, maximum_characters)
    except ModelInputError as error:
        message = str(error)
        field = "content"
        code = "input_too_long" if "maximum supported length" in message else "empty_input"
        raise RequestValidationError(
            message,
            code=code,
            field=field,
            status_code=422,
        ) from error

    return PredictionInput(title=title, article=article)


def _extract_payload(request: Request) -> Mapping[str, Any]:
    if request.is_json:
        payload = request.get_json(silent=True)

        if payload is None:
            raise RequestValidationError(
                "The request body must contain valid JSON.",
                code="invalid_json",
            )

        if not isinstance(payload, dict):
            raise RequestValidationError(
                "The JSON request body must be an object.",
                code="invalid_json_shape",
            )

        return payload

    if request.form:
        return request.form

    raise RequestValidationError(
        "Use application/json or submit HTML form fields.",
        code="unsupported_content_type",
        status_code=415,
    )
