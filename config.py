# config.py
# Defines environment-independent Flask defaults for Fake News Detection.

from __future__ import annotations

from pathlib import Path
from typing import Final


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent
FINAL_ARTIFACT_DIRECTORY: Final[Path] = (
    PROJECT_ROOT / "model_artifacts" / "final"
)


class DefaultConfig:
    """Safe defaults shared by development, testing, and packaged execution."""

    TESTING = False
    DEBUG = False
    JSON_SORT_KEYS = False

    MODEL_PIPELINE_PATH = (
        FINAL_ARTIFACT_DIRECTORY / "fake_news_pipeline.joblib"
    )
    MODEL_METADATA_PATH = (
        FINAL_ARTIFACT_DIRECTORY / "model_metadata.json"
    )
    MODEL_MAX_INPUT_CHARACTERS = 20_000
    MODEL_EAGER_LOAD = False

    MAX_CONTENT_LENGTH = 64 * 1024


class DevelopmentConfig(DefaultConfig):
    """Local Visual Studio development configuration."""

    DEBUG = True


class TestingConfig(DefaultConfig):
    """Deterministic configuration used by automated Flask tests."""

    TESTING = True
    MODEL_EAGER_LOAD = False


class ProductionConfig(DefaultConfig):
    """Deployment-safe configuration without development debugging."""

    DEBUG = False
