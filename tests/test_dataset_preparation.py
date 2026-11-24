# test_dataset_preparation.py
# Verifies deterministic, complete, disjoint, and stratified split manifests.

from __future__ import annotations

import pandas as pd
import pytest

from model.dataset_preparation import (
    LABEL_COLUMN,
    RECORD_ID_COLUMN,
    SPLIT_COLUMN,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    DatasetPreparationError,
    create_record_id,
    create_split_manifest,
    validate_manifest,
    validate_split_configuration,
)
from model.preprocessing import IDENTITY_KEY_COLUMN


def make_cleaned_frame(rows_per_label: int = 100) -> pd.DataFrame:
    labels = [0] * rows_per_label + [1] * rows_per_label
    identities = [f"title-{index}\nbody-{index}" for index in range(len(labels))]
    return pd.DataFrame(
        {
            IDENTITY_KEY_COLUMN: identities,
            LABEL_COLUMN: labels,
        }
    )


def test_create_record_id_is_deterministic_and_content_sensitive() -> None:
    assert create_record_id("alpha") == create_record_id("alpha")
    assert create_record_id("alpha") != create_record_id("beta")
    assert len(create_record_id("alpha")) == 64


def test_validate_split_configuration_accepts_70_15_15() -> None:
    validate_split_configuration(0.70, 0.15, 0.15)


@pytest.mark.parametrize(
    "fractions",
    [
        (0.70, 0.15, 0.14),
        (0.0, 0.50, 0.50),
        (0.70, -0.10, 0.40),
        (1.0, 0.0, 0.0),
    ],
)
def test_validate_split_configuration_rejects_invalid_values(
    fractions: tuple[float, float, float],
) -> None:
    with pytest.raises(DatasetPreparationError):
        validate_split_configuration(*fractions)


def test_create_split_manifest_is_deterministic() -> None:
    cleaned = make_cleaned_frame()
    first = create_split_manifest(cleaned)
    second = create_split_manifest(cleaned)
    pd.testing.assert_frame_equal(first, second)


def test_different_random_state_changes_assignments() -> None:
    cleaned = make_cleaned_frame()
    first = create_split_manifest(cleaned, random_state=205)
    second = create_split_manifest(cleaned, random_state=206)
    merged = first.merge(second, on=RECORD_ID_COLUMN, suffixes=("_first", "_second"))
    assert (merged[f"{SPLIT_COLUMN}_first"] != merged[f"{SPLIT_COLUMN}_second"]).any()


def test_manifest_has_complete_unique_coverage() -> None:
    cleaned = make_cleaned_frame()
    manifest = create_split_manifest(cleaned)
    assert len(manifest) == len(cleaned)
    assert manifest[RECORD_ID_COLUMN].is_unique
    assert set(manifest[SPLIT_COLUMN]) == {
        TRAIN_SPLIT,
        VALIDATION_SPLIT,
        TEST_SPLIT,
    }


def test_manifest_uses_expected_70_15_15_counts() -> None:
    manifest = create_split_manifest(make_cleaned_frame())
    counts = manifest[SPLIT_COLUMN].value_counts().to_dict()
    assert counts == {
        TRAIN_SPLIT: 140,
        VALIDATION_SPLIT: 30,
        TEST_SPLIT: 30,
    }


def test_every_split_contains_equal_synthetic_class_counts() -> None:
    manifest = create_split_manifest(make_cleaned_frame())
    table = pd.crosstab(manifest[SPLIT_COLUMN], manifest[LABEL_COLUMN])
    assert table.loc[TRAIN_SPLIT].to_dict() == {0: 70, 1: 70}
    assert table.loc[VALIDATION_SPLIT].to_dict() == {0: 15, 1: 15}
    assert table.loc[TEST_SPLIT].to_dict() == {0: 15, 1: 15}


def test_create_split_manifest_rejects_duplicate_identity_keys() -> None:
    cleaned = make_cleaned_frame()
    cleaned.loc[1, IDENTITY_KEY_COLUMN] = cleaned.loc[0, IDENTITY_KEY_COLUMN]
    with pytest.raises(DatasetPreparationError, match="Identity keys must be unique"):
        create_split_manifest(cleaned)


def test_create_split_manifest_rejects_missing_required_column() -> None:
    cleaned = make_cleaned_frame().drop(columns=[IDENTITY_KEY_COLUMN])
    with pytest.raises(DatasetPreparationError, match="missing required columns"):
        create_split_manifest(cleaned)


def test_create_split_manifest_rejects_non_binary_labels() -> None:
    cleaned = make_cleaned_frame()
    cleaned.loc[0, LABEL_COLUMN] = 2
    with pytest.raises(DatasetPreparationError, match="labels 0 and 1 only"):
        create_split_manifest(cleaned)


def test_validate_manifest_rejects_duplicate_record_ids() -> None:
    manifest = create_split_manifest(make_cleaned_frame())
    manifest.loc[1, RECORD_ID_COLUMN] = manifest.loc[0, RECORD_ID_COLUMN]
    with pytest.raises(DatasetPreparationError, match="not unique"):
        validate_manifest(manifest, expected_rows=len(manifest))


def test_validate_manifest_rejects_incomplete_coverage() -> None:
    manifest = create_split_manifest(make_cleaned_frame()).iloc[:-1].copy()
    with pytest.raises(DatasetPreparationError, match="does not match"):
        validate_manifest(manifest, expected_rows=200)
