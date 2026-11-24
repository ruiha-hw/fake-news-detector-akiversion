# app.py
# Exposes the Flask application and starts the local development server.

from __future__ import annotations

import os

from flask import Flask

from application import create_app


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5000


def create_runtime_app() -> Flask:
    """Creates the WSGI application used by Visual Studio and Flask tooling."""

    return create_app()


def read_debug_setting() -> bool:
    """Reads an explicit local debug flag without enabling it by default."""

    value = os.environ.get("FAKENEWS_DEBUG", "").strip().casefold()
    return value in {"1", "true", "yes", "on"}


def read_port_setting() -> int:
    """Reads and validates the optional local server port."""

    raw_value = os.environ.get("FAKENEWS_PORT", str(DEFAULT_PORT)).strip()

    try:
        port = int(raw_value)
    except ValueError as error:
        raise RuntimeError("FAKENEWS_PORT must be an integer.") from error

    if port < 1 or port > 65_535:
        raise RuntimeError("FAKENEWS_PORT must be between 1 and 65535.")

    return port


app = create_runtime_app()


if __name__ == "__main__":
    app.run(
        host=DEFAULT_HOST,
        port=read_port_setting(),
        debug=read_debug_setting(),
        use_reloader=read_debug_setting(),
    )
