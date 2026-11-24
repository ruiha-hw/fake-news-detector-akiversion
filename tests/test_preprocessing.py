# test_preprocessing.py
# Verifies WELFake validation, feature creation, cleaning, and deduplication.

from __future__ import annotations

from typing import cast

import pandas as pd
import pytest

from model.preprocessing import (
    ARTICLE_MARKER,
    IDENTITY_KEY_COLUMN,
    LABEL_COLUMN,
    LABEL_FAKE,
    LABEL_REAL,
    MODEL_TEXT_COLUMN,
    SOURCE_INDEX_COLUMN,
    TEXT_COLUMN,
    TITLE_COLUMN,
    TITLE_MARKER,
    DatasetValidationError,
    FeatureMode,
    create_identity_key,
    create_model_text,
    normalise_whitespace,
    preprocess_welfake,
    validate_source_frame,
)


def make_frame() -> pd.DataFrame:
    """Creates a valid frame containing one normalised duplicate pair."""

    return pd.DataFrame(
        {
            SOURCE_INDEX_COLUMN: [
                10,
                11,
                12,
                13,
            ],
            TITLE_COLUMN: [
                " Real headline ",
                "Fake headline",
                None,
                " real headline",
            ],
            TEXT_COLUMN: [
                "Real body",
                "Fake body",
                "Body only",
                "Real   body",
            ],
            LABEL_COLUMN: [
                LABEL_REAL,
                LABEL_FAKE,
                LABEL_REAL,
                LABEL_REAL,
            ],
        }
    )


def test_normalise_whitespace_handles_missing_values() -> None:
    assert normalise_whitespace(None) == ""
    assert normalise_whitespace(float("nan")) == ""


def test_normalise_whitespace_collapses_spacing() -> None:
    value = "  alpha\n\tbeta   gamma  "

    result = normalise_whitespace(value)

    assert result == "alpha beta gamma"


def test_normalise_whitespace_preserves_case() -> None:
    result = normalise_whitespace(
        "  Mixed CASE Text  "
    )

    assert result == "Mixed CASE Text"


def test_create_identity_key_is_case_insensitive() -> None:
    first = create_identity_key(
        " Headline ",
        "Article body",
    )

    second = create_identity_key(
        "headline",
        "article body",
    )

    assert first == second


def test_create_identity_key_is_whitespace_insensitive() -> None:
    first = create_identity_key(
        "Headline",
        "Article   body",
    )

    second = create_identity_key(
        " Headline ",
        "Article body",
    )

    assert first == second


@pytest.mark.parametrize(
    (
        "mode",
        "expected",
    ),
    [
        (
            "title",
            "Headline",
        ),
        (
            "text",
            "Article body",
        ),
        (
            "combined",
            (
                f"{TITLE_MARKER}\n"
                "Headline\n\n"
                f"{ARTICLE_MARKER}\n"
                "Article body"
            ),
        ),
    ],
)
def test_create_model_text_supports_each_feature_mode(
    mode: FeatureMode,
    expected: str,
) -> None:
    result = create_model_text(
        " Headline ",
        "Article   body",
        mode,
    )

    assert result == expected


def test_create_model_text_supports_empty_title() -> None:
    result = create_model_text(
        None,
        "Article body",
        "combined",
    )

    assert result == (
        f"{TITLE_MARKER}\n"
        "\n\n"
        f"{ARTICLE_MARKER}\n"
        "Article body"
    )


def test_create_model_text_supports_empty_body() -> None:
    result = create_model_text(
        "Headline",
        None,
        "combined",
    )

    assert result == (
        f"{TITLE_MARKER}\n"
        "Headline\n\n"
        f"{ARTICLE_MARKER}\n"
    )


def test_create_model_text_rejects_unknown_mode() -> None:
    invalid_mode = cast(
        FeatureMode,
        "unknown",
    )

    with pytest.raises(
        ValueError,
        match="Unsupported feature mode",
    ):
        create_model_text(
            "Headline",
            "Body",
            invalid_mode,
        )


@pytest.mark.parametrize(
    "missing_column",
    [
        TITLE_COLUMN,
        TEXT_COLUMN,
        LABEL_COLUMN,
    ],
)
def test_validate_source_frame_rejects_missing_column(
    missing_column: str,
) -> None:
    frame = make_frame().drop(
        columns=[missing_column]
    )

    with pytest.raises(
        DatasetValidationError,
        match="Required columns are missing",
    ):
        validate_source_frame(frame)


def test_validate_source_frame_rejects_non_numeric_label() -> None:
    frame = make_frame()
    frame[LABEL_COLUMN] = frame[
        LABEL_COLUMN
    ].astype("object")

    frame.loc[
        0,
        LABEL_COLUMN,
    ] = "not-a-label"

    with pytest.raises(
        DatasetValidationError,
        match="missing or non-numeric",
    ):
        validate_source_frame(frame)


def test_validate_source_frame_rejects_missing_label() -> None:
    frame = make_frame()

    frame.loc[
        0,
        LABEL_COLUMN,
    ] = None

    with pytest.raises(
        DatasetValidationError,
        match="missing or non-numeric",
    ):
        validate_source_frame(frame)


def test_validate_source_frame_rejects_unexpected_label() -> None:
    frame = make_frame()

    frame.loc[
        0,
        LABEL_COLUMN,
    ] = 2

    with pytest.raises(
        DatasetValidationError,
        match="Unexpected labels detected",
    ):
        validate_source_frame(frame)


def test_validate_source_frame_accepts_binary_labels() -> None:
    validate_source_frame(
        make_frame()
    )


def test_preprocess_does_not_mutate_source_frame() -> None:
    source = make_frame()
    original = source.copy(
        deep=True
    )

    preprocess_welfake(source)

    pd.testing.assert_frame_equal(
        source,
        original,
    )


def test_preprocess_removes_source_index_column() -> None:
    cleaned, _ = preprocess_welfake(
        make_frame()
    )

    assert SOURCE_INDEX_COLUMN not in cleaned.columns


def test_preprocess_creates_required_output_columns() -> None:
    cleaned, _ = preprocess_welfake(
        make_frame()
    )

    expected_columns = {
        TITLE_COLUMN,
        TEXT_COLUMN,
        LABEL_COLUMN,
        IDENTITY_KEY_COLUMN,
        MODEL_TEXT_COLUMN,
    }

    assert expected_columns.issubset(
        cleaned.columns
    )


def test_preprocess_removes_normalised_pair_duplicate() -> None:
    cleaned, summary = preprocess_welfake(
        make_frame()
    )

    assert len(cleaned) == 3
    assert summary.source_rows == 4
    assert summary.output_rows == 3
    assert summary.removed_duplicate_rows == 1
    assert cleaned[
        IDENTITY_KEY_COLUMN
    ].is_unique


def test_preprocess_preserves_first_duplicate_occurrence() -> None:
    cleaned, _ = preprocess_welfake(
        make_frame()
    )

    assert cleaned.loc[
        0,
        TITLE_COLUMN,
    ] == "Real headline"

    assert cleaned.loc[
        0,
        TEXT_COLUMN,
    ] == "Real body"


def test_preprocess_retains_title_only_record() -> None:
    frame = pd.DataFrame(
        {
            TITLE_COLUMN: [
                "Title only",
                "Fake title",
            ],
            TEXT_COLUMN: [
                "",
                "Fake body",
            ],
            LABEL_COLUMN: [
                LABEL_REAL,
                LABEL_FAKE,
            ],
        }
    )

    cleaned, summary = preprocess_welfake(
        frame
    )

    assert len(cleaned) == 2
    assert summary.removed_unusable_rows == 0

    assert cleaned.loc[
        0,
        TITLE_COLUMN,
    ] == "Title only"

    assert cleaned.loc[
        0,
        TEXT_COLUMN,
    ] == ""


def test_preprocess_retains_body_only_record() -> None:
    frame = pd.DataFrame(
        {
            TITLE_COLUMN: [
                "",
                "Fake title",
            ],
            TEXT_COLUMN: [
                "Body only",
                "Fake body",
            ],
            LABEL_COLUMN: [
                LABEL_REAL,
                LABEL_FAKE,
            ],
        }
    )

    cleaned, summary = preprocess_welfake(
        frame
    )

    assert len(cleaned) == 2
    assert summary.removed_unusable_rows == 0

    assert cleaned.loc[
        0,
        TITLE_COLUMN,
    ] == ""

    assert cleaned.loc[
        0,
        TEXT_COLUMN,
    ] == "Body only"


def test_preprocess_removes_fully_empty_record() -> None:
    frame = pd.DataFrame(
        {
            TITLE_COLUMN: [
                "Title only",
                "   ",
                "Fake title",
            ],
            TEXT_COLUMN: [
                "",
                "\n\t",
                "Fake body",
            ],
            LABEL_COLUMN: [
                LABEL_REAL,
                LABEL_REAL,
                LABEL_FAKE,
            ],
        }
    )

    cleaned, summary = preprocess_welfake(
        frame
    )

    assert len(cleaned) == 2
    assert summary.removed_unusable_rows == 1


def test_preprocess_rejects_conflicting_pair_labels() -> None:
    frame = pd.DataFrame(
        {
            TITLE_COLUMN: [
                "Same title",
                " same title ",
            ],
            TEXT_COLUMN: [
                "Same body",
                "same   body",
            ],
            LABEL_COLUMN: [
                LABEL_FAKE,
                LABEL_REAL,
            ],
        }
    )

    with pytest.raises(
        DatasetValidationError,
        match="Conflicting labels",
    ):
        preprocess_welfake(frame)


def test_preprocess_creates_combined_model_text() -> None:
    cleaned, _ = preprocess_welfake(
        make_frame(),
        mode="combined",
    )

    expected = (
        f"{TITLE_MARKER}\n"
        "Real headline\n\n"
        f"{ARTICLE_MARKER}\n"
        "Real body"
    )

    assert cleaned.loc[
        0,
        MODEL_TEXT_COLUMN,
    ] == expected


@pytest.mark.parametrize(
    (
        "mode",
        "expected",
    ),
    [
        (
            "title",
            "Real headline",
        ),
        (
            "text",
            "Real body",
        ),
    ],
)
def test_preprocess_creates_selected_feature_text(
    mode: FeatureMode,
    expected: str,
) -> None:
    cleaned, _ = preprocess_welfake(
        make_frame(),
        mode=mode,
    )

    assert cleaned.loc[
        0,
        MODEL_TEXT_COLUMN,
    ] == expected


def test_preprocess_converts_labels_to_int8() -> None:
    cleaned, _ = preprocess_welfake(
        make_frame()
    )

    assert str(
        cleaned[LABEL_COLUMN].dtype
    ) == "int8"


def test_preprocessing_summary_reports_class_counts() -> None:
    _, summary = preprocess_welfake(
        make_frame()
    )

    assert summary.fake_rows == 1
    assert summary.real_rows == 2


def test_preprocessing_summary_serialises_to_dictionary() -> None:
    _, summary = preprocess_welfake(
        make_frame()
    )

    assert summary.to_dict() == {
        "source_rows": 4,
        "output_rows": 3,
        "removed_duplicate_rows": 1,
        "removed_unusable_rows": 0,
        "fake_rows": 1,
        "real_rows": 2,
    }