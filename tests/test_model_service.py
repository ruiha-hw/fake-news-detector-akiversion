# test_model_service.py
# Verifies model integrity checks, runtime input rules, and prediction mapping.

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from model.model_service import (
    MAX_INPUT_CHARACTERS,
    ModelInputError,
    ModelIntegrityError,
    ModelService,
    calculate_sha256,
    prepare_runtime_input,
)


def build_test_pipeline() -> Pipeline:
    texts = [
        "TITLE:\nfalse rumour\n\nARTICLE:\nfabricated hoax claim",
        "TITLE:\nmisleading story\n\nARTICLE:\ninvented false report",
        "TITLE:\nverified report\n\nARTICLE:\nofficial confirmed evidence",
        "TITLE:\nfactual update\n\nARTICLE:\nconfirmed public record",
    ]
    labels = np.array([0, 0, 1, 1])
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            (
                "classifier",
                LogisticRegression(solver="liblinear", random_state=205),
            ),
        ]
    )
    pipeline.fit(texts, labels)
    return pipeline


def write_artifacts(directory: Path) -> tuple[Path, Path]:
    pipeline_path = directory / "pipeline.joblib"
    metadata_path = directory / "metadata.json"
    joblib.dump(build_test_pipeline(), pipeline_path)
    metadata_path.write_text(
        json.dumps(
            {
                "pipeline_sha256": calculate_sha256(pipeline_path),
                "feature_mode": "combined",
                "label_mapping": {"0": "likely fake", "1": "likely real"},
            }
        ),
        encoding="utf-8",
    )
    return pipeline_path, metadata_path


def test_prepare_runtime_input_builds_locked_combined_format() -> None:
    value, count = prepare_runtime_input(" Headline ", "Article   body")
    assert value == "TITLE:\nHeadline\n\nARTICLE:\nArticle body"
    assert count == len("Headline") + len("Article body")


@pytest.mark.parametrize("title, article", [("", ""), ("   ", "\n\t")])
def test_prepare_runtime_input_rejects_empty_content(title: str, article: str) -> None:
    with pytest.raises(ModelInputError, match="Enter a headline"):
        prepare_runtime_input(title, article)


def test_prepare_runtime_input_rejects_non_string_values() -> None:
    with pytest.raises(ModelInputError, match="must both be strings"):
        prepare_runtime_input(123, "article")


def test_prepare_runtime_input_rejects_excessive_length() -> None:
    with pytest.raises(ModelInputError, match="maximum supported length"):
        prepare_runtime_input("x" * (MAX_INPUT_CHARACTERS + 1), "")


def test_service_loads_verified_pipeline_lazily(tmp_path: Path) -> None:
    pipeline_path, metadata_path = write_artifacts(tmp_path)
    service = ModelService(pipeline_path, metadata_path)
    assert service.is_loaded is False
    service.load()
    assert service.is_loaded is True


def test_service_load_is_idempotent(tmp_path: Path) -> None:
    pipeline_path, metadata_path = write_artifacts(tmp_path)
    service = ModelService(pipeline_path, metadata_path)
    service.load()
    service.load()
    assert service.is_loaded is True


def test_service_rejects_pipeline_hash_mismatch(tmp_path: Path) -> None:
    pipeline_path, metadata_path = write_artifacts(tmp_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["pipeline_sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ModelIntegrityError, match="SHA-256 mismatch"):
        ModelService(pipeline_path, metadata_path).load()


def test_service_rejects_wrong_feature_mode(tmp_path: Path) -> None:
    pipeline_path, metadata_path = write_artifacts(tmp_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["feature_mode"] = "text"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ModelIntegrityError, match="combined feature mode"):
        ModelService(pipeline_path, metadata_path).load()


def test_service_rejects_missing_pipeline(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps({"pipeline_sha256": "0" * 64, "feature_mode": "combined"}),
        encoding="utf-8",
    )
    with pytest.raises(ModelIntegrityError, match="pipeline was not found"):
        ModelService(tmp_path / "missing.joblib", metadata_path).load()


def test_predict_returns_serialisable_probability_result(tmp_path: Path) -> None:
    pipeline_path, metadata_path = write_artifacts(tmp_path)
    result = ModelService(pipeline_path, metadata_path).predict(
        "verified report",
        "official confirmed evidence",
    )
    payload = result.to_dict()
    assert payload["label"] in {0, 1}
    assert payload["classification"] in {"seek further information", "potentially misleading", "more consistent with real-news patterns"}
    assert 0.0 <= payload["confidence"] <= 1.0
    assert payload["fake_probability"] + payload["real_probability"] == pytest.approx(
        1.0,
        abs=1e-7,
    )
    assert payload["confidence"] == max(
        payload["fake_probability"],
        payload["real_probability"],
    )


def test_metadata_returns_copy(tmp_path: Path) -> None:
    pipeline_path, metadata_path = write_artifacts(tmp_path)
    service = ModelService(pipeline_path, metadata_path)
    first = service.metadata
    first["feature_mode"] = "changed"
    assert service.metadata["feature_mode"] == "combined"
