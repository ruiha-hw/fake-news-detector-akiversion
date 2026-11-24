# preprocessing.py
# Defines rules for WELFake cleaning, identity, feature, and deduplication.

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final, Literal

import pandas as pd


TITLE_COLUMN: Final[str] = "title"
TEXT_COLUMN: Final[str] = "text"
LABEL_COLUMN: Final[str] = "label"
SOURCE_INDEX_COLUMN: Final[str] = "Unnamed: 0"

MODEL_TEXT_COLUMN: Final[str] = "model_text"
IDENTITY_KEY_COLUMN: Final[str] = "identity_key"

LABEL_FAKE: Final[int] = 0
LABEL_REAL: Final[int] = 1

VALID_LABELS: Final[frozenset[int]] = frozenset(
    {
        LABEL_FAKE,
        LABEL_REAL,
    }
)

TITLE_MARKER: Final[str] = "TITLE:"
ARTICLE_MARKER: Final[str] = "ARTICLE:"

FeatureMode = Literal[
    "title",
    "text",
    "combined",
]

_WHITESPACE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\s+"
)


class DatasetValidationError(ValueError):
    """Raised when source data violates the required WELFake contract."""


@dataclass(frozen=True)
class PreprocessingSummary:
    """Records deterministic preprocessing outcomes."""

    source_rows: int
    output_rows: int
    removed_duplicate_rows: int
    removed_unusable_rows: int
    fake_rows: int
    real_rows: int

    def to_dict(self) -> dict[str, int]:
        return {
            "source_rows": self.source_rows,
            "output_rows": self.output_rows,
            "removed_duplicate_rows": (
                self.removed_duplicate_rows
            ),
            "removed_unusable_rows": (
                self.removed_unusable_rows
            ),
            "fake_rows": self.fake_rows,
            "real_rows": self.real_rows,
        }


def normalise_whitespace(value: object) -> str:
    """Converts missing values to empty text and collapses whitespace."""

    if pd.isna(value):
        return ""

    return _WHITESPACE_PATTERN.sub(
        " ",
        str(value),
    ).strip()


def create_identity_component(value: object) -> str:
    """Creates a case-insensitive deduplication component."""

    return normalise_whitespace(value).casefold()


def create_identity_key(
    title: object,
    text: object,
) -> str:
    """Creates the title/body identity used for deduplication."""

    return (
        f"{create_identity_component(title)}\n"
        f"{create_identity_component(text)}"
    )


def create_model_text(
    title: object,
    text: object,
    mode: FeatureMode = "combined",
) -> str:
    """Creates stable model input without destructive text cleaning."""

    clean_title = normalise_whitespace(title)
    clean_text = normalise_whitespace(text)

    if mode == "title":
        return clean_title

    if mode == "text":
        return clean_text

    if mode == "combined":
        return (
            f"{TITLE_MARKER}\n"
            f"{clean_title}\n\n"
            f"{ARTICLE_MARKER}\n"
            f"{clean_text}"
        )

    raise ValueError(
        f"Unsupported feature mode: {mode!r}"
    )


def validate_source_frame(
    frame: pd.DataFrame,
) -> None:
    """Validates required columns and binary labels."""

    required_columns = {
        TITLE_COLUMN,
        TEXT_COLUMN,
        LABEL_COLUMN,
    }

    missing_columns = required_columns.difference(
        frame.columns
    )

    if missing_columns:
        missing = ", ".join(
            sorted(missing_columns)
        )

        raise DatasetValidationError(
            f"Required columns are missing: {missing}"
        )

    numeric_labels = pd.to_numeric(
        frame[LABEL_COLUMN],
        errors="coerce",
    )

    if numeric_labels.isna().any():
        invalid_count = int(
            numeric_labels.isna().sum()
        )

        raise DatasetValidationError(
            "The label column contains "
            f"{invalid_count} missing or "
            "non-numeric values."
        )

    observed_labels = {
        int(value)
        for value in numeric_labels.unique()
    }

    if observed_labels != VALID_LABELS:
        raise DatasetValidationError(
            "Unexpected labels detected. "
            f"Expected {sorted(VALID_LABELS)}, "
            f"found {sorted(observed_labels)}."
        )


def preprocess_welfake(
    frame: pd.DataFrame,
    mode: FeatureMode = "combined",
) -> tuple[
    pd.DataFrame,
    PreprocessingSummary,
]:
    """Cleans and deduplicates WELFake without mutating source data."""

    validate_source_frame(frame)

    source_rows = len(frame)
    working = frame.copy(deep=True)

    if SOURCE_INDEX_COLUMN in working.columns:
        working = working.drop(
            columns=[SOURCE_INDEX_COLUMN]
        )

    working[TITLE_COLUMN] = working[
        TITLE_COLUMN
    ].map(normalise_whitespace)

    working[TEXT_COLUMN] = working[
        TEXT_COLUMN
    ].map(normalise_whitespace)

    working[LABEL_COLUMN] = pd.to_numeric(
        working[LABEL_COLUMN],
        errors="raise",
    ).astype("int8")

    usable_mask = (
        working[TITLE_COLUMN].ne("")
        | working[TEXT_COLUMN].ne("")
    )

    removed_unusable_rows = int(
        (~usable_mask).sum()
    )

    working = working.loc[
        usable_mask
    ].copy()

    working[IDENTITY_KEY_COLUMN] = [
        create_identity_key(
            title,
            text,
        )
        for title, text in zip(
            working[TITLE_COLUMN],
            working[TEXT_COLUMN],
            strict=True,
        )
    ]

    conflict_counts = (
        working
        .groupby(
            IDENTITY_KEY_COLUMN
        )[LABEL_COLUMN]
        .nunique()
    )

    conflicting_keys = conflict_counts[
        conflict_counts.gt(1)
    ]

    if not conflicting_keys.empty:
        raise DatasetValidationError(
            "Conflicting labels were found for "
            f"{len(conflicting_keys)} identical "
            "title/body identities."
        )

    rows_before_deduplication = len(
        working
    )

    working = working.drop_duplicates(
        subset=[IDENTITY_KEY_COLUMN],
        keep="first",
    ).copy()

    removed_duplicate_rows = (
        rows_before_deduplication
        - len(working)
    )

    working[MODEL_TEXT_COLUMN] = [
        create_model_text(
            title,
            text,
            mode,
        )
        for title, text in zip(
            working[TITLE_COLUMN],
            working[TEXT_COLUMN],
            strict=True,
        )
    ]

    working = working.reset_index(
        drop=True
    )

    fake_rows = int(
        working[LABEL_COLUMN]
        .eq(LABEL_FAKE)
        .sum()
    )

    real_rows = int(
        working[LABEL_COLUMN]
        .eq(LABEL_REAL)
        .sum()
    )

    summary = PreprocessingSummary(
        source_rows=source_rows,
        output_rows=len(working),
        removed_duplicate_rows=(
            removed_duplicate_rows
        ),
        removed_unusable_rows=(
            removed_unusable_rows
        ),
        fake_rows=fake_rows,
        real_rows=real_rows,
    )

    return working, summary