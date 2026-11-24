# test_frontend_ui.py
# Verifies Dark Glass templates, static assets, and classifier integration hooks.

from __future__ import annotations

from application import create_app


def client():
    return create_app({"TESTING": True}).test_client()


def test_home_renders_dark_glass_classifier_structure() -> None:
    response = client().get("/")
    html = response.get_data(as_text=True)
    assert response.status_code == 200
    assert 'id="prediction-form"' in html
    assert 'id="result-card"' in html
    assert 'data-max-characters="20000"' in html
    assert "Question the certainty" in html


def test_home_links_required_static_assets() -> None:
    html = client().get("/").get_data(as_text=True)
    assert '/static/css/style.css' in html
    assert '/static/js/application.js' in html


def test_about_renders_method_and_limitation_content() -> None:
    html = client().get("/about").get_data(as_text=True)
    assert "How the model was built" in html
    assert "Classification is not verification" in html
    assert "96.42%" in html


def test_stylesheet_is_available_and_contains_glass_tokens() -> None:
    response = client().get("/static/css/style.css")
    css = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "--surface-1" in css
    assert "backdrop-filter" in css
    assert "prefers-reduced-motion" in css


def test_javascript_is_available_and_targets_prediction_endpoint() -> None:
    response = client().get("/static/js/application.js")
    script = response.get_data(as_text=True)
    assert response.status_code == 200
    assert 'fetch(form.action' in script
    assert 'application/json' in script
    assert 'checkHealth' in script


def test_home_contains_accessibility_landmarks() -> None:
    html = client().get("/").get_data(as_text=True)
    assert 'class="skip-link"' in html
    assert 'id="main-content"' in html
    assert 'aria-live="polite"' in html
    assert 'role="alert"' in html


def test_ui_displays_model_limitations() -> None:
    html = client().get("/").get_data(as_text=True)
    assert "not factual verification" in html
    assert "Verify important claims" in html
