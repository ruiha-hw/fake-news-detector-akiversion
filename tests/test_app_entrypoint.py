# test_app_entrypoint.py
# Verifies the executable and WSGI entry-point contract.

from __future__ import annotations

import importlib

import pytest
from flask import Flask

import app as app_module


def test_module_exposes_flask_application() -> None:
    assert isinstance(app_module.app, Flask)


def test_runtime_app_contains_expected_routes() -> None:
    runtime_app = app_module.create_runtime_app()
    rules = {str(rule) for rule in runtime_app.url_map.iter_rules()}
    assert {"/", "/about", "/health", "/predict"}.issubset(rules)


def test_debug_setting_defaults_to_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FAKENEWS_DEBUG", raising=False)
    assert app_module.read_debug_setting() is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_debug_setting_accepts_explicit_true_values(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv("FAKENEWS_DEBUG", value)
    assert app_module.read_debug_setting() is True


def test_port_setting_defaults_to_5000(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FAKENEWS_PORT", raising=False)
    assert app_module.read_port_setting() == 5000


def test_port_setting_accepts_valid_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKENEWS_PORT", "5055")
    assert app_module.read_port_setting() == 5055


@pytest.mark.parametrize("value", ["invalid", "0", "65536", "-1"])
def test_port_setting_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv("FAKENEWS_PORT", value)
    with pytest.raises(RuntimeError, match="FAKENEWS_PORT"):
        app_module.read_port_setting()


def test_import_does_not_eagerly_load_model() -> None:
    reloaded = importlib.reload(app_module)
    assert reloaded.app.extensions["model_service"].is_loaded is False
