# model_service.py
# Verifies, loads, and interprets the final WELFake classification pipe.

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from model.preprocessing import create_model_text, normalise_whitespace

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DEFAULT_PIPELINE_PATH: Final[Path] = PROJECT_ROOT / "model_artifacts" / "final" / "fake_news_pipeline.joblib"
DEFAULT_METADATA_PATH: Final[Path] = PROJECT_ROOT / "model_artifacts" / "final" / "model_metadata.json"
EXPECTED_STEPS: Final[tuple[str, str]] = ("tfidf", "classifier")
EXPECTED_CLASSES: Final[tuple[int, int]] = (0, 1)
EXPECTED_FEATURE_MODE: Final[str] = "combined"
MAX_INPUT_CHARACTERS: Final[int] = 20_000
INTERPRETATION_THRESHOLD: Final[float] = 0.75

MODEL_LABEL_DISPLAY: Final[dict[int, str]] = {0: "likely fake", 1: "likely real"}
INTERPRETATION_DISPLAY: Final[dict[int, str]] = {
    0: "potentially misleading",
    1: "more consistent with real-news patterns",
}
INTERPRETATION_GUIDANCE: Final[dict[int, str]] = {
    0: "This text more strongly matches patterns associated with fake-labelled articles. Verify its claims before relying on or sharing it.",
    1: "This text more strongly matches patterns associated with real-labelled articles. This classification does not confirm that its claims are true.",
}
UNCERTAIN_CLASSIFICATION: Final[str] = "seek further information"
UNCERTAIN_GUIDANCE: Final[str] = "The model does not strongly favour either learned category. Seek confirmation from reliable sources before treating this content as accurate."


class ModelServiceError(RuntimeError):
    """Base exception for model-service failures."""


class ModelIntegrityError(ModelServiceError):
    """Raised when an artefact or its metadata fails integrity validation."""


class ModelInputError(ModelServiceError, ValueError):
    """Raised when prediction input violates the runtime contract."""


@dataclass(frozen=True)
class PredictionResult:
    """Serializable raw model evidence plus audited product interpretation."""

    label: int
    classification: str
    confidence: float
    fake_probability: float
    real_probability: float
    input_characters: int
    decisive: bool = True
    interpretation_threshold: float = INTERPRETATION_THRESHOLD
    guidance: str = ""
    model_classification: str = ""

    def to_dict(self) -> dict[str, int | str | float | bool]:
        return asdict(self)


def calculate_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_runtime_input(title: object = "", article: object = "", maximum_characters: int = MAX_INPUT_CHARACTERS) -> tuple[str, int]:
    if not isinstance(title, str) or not isinstance(article, str):
        raise ModelInputError("Title and article must both be strings.")
    clean_title = normalise_whitespace(title)
    clean_article = normalise_whitespace(article)
    if not clean_title and not clean_article:
        raise ModelInputError("Enter a headline, article text, or both.")
    input_characters = len(clean_title) + len(clean_article)
    if input_characters > maximum_characters:
        raise ModelInputError(f"Input exceeds the maximum supported length of {maximum_characters:,} characters.")
    return create_model_text(clean_title, clean_article, "combined"), input_characters


def interpret_prediction(label: int, confidence: float, threshold: float = INTERPRETATION_THRESHOLD) -> tuple[bool, str, str]:
    """Applies the validation-audited 0.75 interpretation policy."""

    if label not in MODEL_LABEL_DISPLAY:
        raise ModelIntegrityError(f"Unsupported model label: {label}")
    if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ModelIntegrityError("Model confidence must be finite and inside [0, 1].")
    if not 0.5 <= threshold <= 1.0:
        raise ModelIntegrityError("Interpretation threshold must be between 0.5 and 1.0.")
    if confidence < threshold:
        return False, UNCERTAIN_CLASSIFICATION, UNCERTAIN_GUIDANCE
    return True, INTERPRETATION_DISPLAY[label], INTERPRETATION_GUIDANCE[label]


class ModelService:
    """Thread-safe lazy loader and prediction facade for the final pipeline."""

    def __init__(self, pipeline_path: Path = DEFAULT_PIPELINE_PATH, metadata_path: Path = DEFAULT_METADATA_PATH, maximum_characters: int = MAX_INPUT_CHARACTERS) -> None:
        self._pipeline_path = Path(pipeline_path)
        self._metadata_path = Path(metadata_path)
        self._maximum_characters = maximum_characters
        self._pipeline: Pipeline | None = None
        self._metadata: dict[str, object] | None = None
        self._load_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    @property
    def metadata(self) -> dict[str, object]:
        self.load()
        assert self._metadata is not None
        return dict(self._metadata)

    def load(self) -> None:
        if self._pipeline is not None:
            return
        with self._load_lock:
            if self._pipeline is not None:
                return
            if not self._metadata_path.is_file():
                raise ModelIntegrityError(f"Model metadata was not found: {self._metadata_path}")
            if not self._pipeline_path.is_file():
                raise ModelIntegrityError(f"Model pipeline was not found: {self._pipeline_path}")
            metadata = json.loads(self._metadata_path.read_text(encoding="utf-8"))
            expected_hash = str(metadata.get("pipeline_sha256", ""))
            if not expected_hash:
                raise ModelIntegrityError("Model metadata does not contain pipeline_sha256.")
            actual_hash = calculate_sha256(self._pipeline_path)
            if actual_hash != expected_hash:
                raise ModelIntegrityError(f"Model pipeline SHA-256 mismatch. Expected {expected_hash}, found {actual_hash}.")
            if metadata.get("feature_mode") != EXPECTED_FEATURE_MODE:
                raise ModelIntegrityError("Model metadata does not specify the locked combined feature mode.")
            pipeline = joblib.load(self._pipeline_path)
            self._validate_pipeline(pipeline)
            self._metadata = metadata
            self._pipeline = pipeline

    def predict(self, title: object = "", article: object = "") -> PredictionResult:
        model_input, input_characters = prepare_runtime_input(title, article, self._maximum_characters)
        self.load()
        assert self._pipeline is not None
        prediction = int(self._pipeline.predict([model_input])[0])
        probabilities = self._pipeline.predict_proba([model_input])[0]
        classes = [int(value) for value in self._pipeline.classes_.tolist()]
        probability_by_class = {label: float(probability) for label, probability in zip(classes, probabilities, strict=True)}
        if prediction not in MODEL_LABEL_DISPLAY:
            raise ModelIntegrityError(f"Pipeline returned unsupported label: {prediction}")
        fake_probability = probability_by_class[0]
        real_probability = probability_by_class[1]
        confidence = probability_by_class[prediction]
        self._validate_probabilities(fake_probability, real_probability, confidence)
        decisive, classification, guidance = interpret_prediction(prediction, confidence)
        return PredictionResult(
            label=prediction,
            classification=classification,
            confidence=round(confidence, 8),
            fake_probability=round(fake_probability, 8),
            real_probability=round(real_probability, 8),
            input_characters=input_characters,
            decisive=decisive,
            interpretation_threshold=INTERPRETATION_THRESHOLD,
            guidance=guidance,
            model_classification=MODEL_LABEL_DISPLAY[prediction],
        )

    @staticmethod
    def _validate_pipeline(pipeline: object) -> None:
        if not isinstance(pipeline, Pipeline):
            raise ModelIntegrityError("Loaded artefact is not a Scikit-learn Pipeline.")
        if tuple(pipeline.named_steps.keys()) != EXPECTED_STEPS:
            raise ModelIntegrityError("Pipeline steps do not match the required tfidf/classifier sequence.")
        if not hasattr(pipeline, "predict") or not hasattr(pipeline, "predict_proba"):
            raise ModelIntegrityError("Pipeline does not support prediction probabilities.")
        classes = tuple(int(value) for value in pipeline.classes_.tolist())
        if classes != EXPECTED_CLASSES:
            raise ModelIntegrityError(f"Pipeline classes must be {EXPECTED_CLASSES}, found {classes}.")

    @staticmethod
    def _validate_probabilities(fake_probability: float, real_probability: float, confidence: float) -> None:
        values = np.array([fake_probability, real_probability, confidence], dtype=float)
        if not np.isfinite(values).all():
            raise ModelIntegrityError("Pipeline returned non-finite probabilities.")
        if not np.logical_and(values >= 0.0, values <= 1.0).all():
            raise ModelIntegrityError("Pipeline returned probabilities outside [0, 1].")
        if not np.isclose(fake_probability + real_probability, 1.0, atol=1e-8):
            raise ModelIntegrityError("Class probabilities do not sum to one.")
