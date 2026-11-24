
# dataset_preparation.py
# Creates deterministic stratified WELFake train, validation, and test manifests.

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Final

import pandas as pd
from sklearn.model_selection import train_test_split

from model.preprocessing import (
    IDENTITY_KEY_COLUMN,
    LABEL_COLUMN,
    preprocess_welfake,
)


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
SOURCE_DATASET: Final[Path] = (
    PROJECT_ROOT / "data" / "raw" / "WELFake_Dataset.csv"
)
PROCESSED_DIRECTORY: Final[Path] = PROJECT_ROOT / "data" / "processed"
MANIFEST_PATH: Final[Path] = PROCESSED_DIRECTORY / "split_manifest.csv"
METADATA_PATH: Final[Path] = (
    PROCESSED_DIRECTORY / "dataset_preparation_metadata.json"
)

SOURCE_MD5: Final[str] = "73c9675a4b3d09f86a6933d0b8d7d908"
RANDOM_STATE: Final[int] = 205
TRAIN_FRACTION: Final[float] = 0.70
VALIDATION_FRACTION: Final[float] = 0.15
TEST_FRACTION: Final[float] = 0.15

RECORD_ID_COLUMN: Final[str] = "record_id"
SPLIT_COLUMN: Final[str] = "split"
TRAIN_SPLIT: Final[str] = "train"
VALIDATION_SPLIT: Final[str] = "validation"
TEST_SPLIT: Final[str] = "test"


class DatasetPreparationError(ValueError):
    """Raised when deterministic split preparation cannot be completed safely."""


def calculate_file_hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)

    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def create_record_id(identity_key: str) -> str:
    """Creates a stable identifier without storing article content in the manifest."""

    return hashlib.sha256(identity_key.encode("utf-8")).hexdigest()


def validate_split_configuration(
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
) -> None:
    fractions = (train_fraction, validation_fraction, test_fraction)

    if any(value <= 0.0 or value >= 1.0 for value in fractions):
        raise DatasetPreparationError("Every split fraction must be between 0 and 1.")

    if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise DatasetPreparationError("Train, validation, and test fractions must total 1.0.")


def _validate_clean_frame(frame: pd.DataFrame) -> None:
    required = {IDENTITY_KEY_COLUMN, LABEL_COLUMN}
    missing = required.difference(frame.columns)

    if missing:
        raise DatasetPreparationError(
            f"Cleaned data is missing required columns: {', '.join(sorted(missing))}"
        )

    if frame.empty:
        raise DatasetPreparationError("Cleaned data contains no records.")

    if frame[IDENTITY_KEY_COLUMN].duplicated().any():
        raise DatasetPreparationError("Identity keys must be unique before splitting.")

    label_counts = frame[LABEL_COLUMN].value_counts()

    if set(label_counts.index.tolist()) != {0, 1}:
        raise DatasetPreparationError("Cleaned data must contain labels 0 and 1 only.")

    if int(label_counts.min()) < 3:
        raise DatasetPreparationError(
            "Each label requires at least three records for three-way stratification."
        )


def create_split_manifest(
    cleaned: pd.DataFrame,
    random_state: int = RANDOM_STATE,
    train_fraction: float = TRAIN_FRACTION,
    validation_fraction: float = VALIDATION_FRACTION,
    test_fraction: float = TEST_FRACTION,
) -> pd.DataFrame:
    """Returns one deterministic split assignment per cleaned dataset identity."""

    validate_split_configuration(
        train_fraction,
        validation_fraction,
        test_fraction,
    )
    _validate_clean_frame(cleaned)

    records = pd.DataFrame(
        {
            RECORD_ID_COLUMN: cleaned[IDENTITY_KEY_COLUMN].map(create_record_id),
            LABEL_COLUMN: cleaned[LABEL_COLUMN].astype("int8"),
        }
    )

    if records[RECORD_ID_COLUMN].duplicated().any():
        raise DatasetPreparationError("A record identifier collision was detected.")

    total_rows = len(records)
    test_rows = round(total_rows * test_fraction)
    validation_rows = round(total_rows * validation_fraction)
    train_rows = total_rows - test_rows - validation_rows

    if min(train_rows, validation_rows, test_rows) < 2:
        raise DatasetPreparationError(
            "The requested fractions do not provide enough rows for every split."
        )

    train_validation, test = train_test_split(
        records,
        test_size=test_rows,
        random_state=random_state,
        shuffle=True,
        stratify=records[LABEL_COLUMN],
    )

    train, validation = train_test_split(
        train_validation,
        test_size=validation_rows,
        random_state=random_state,
        shuffle=True,
        stratify=train_validation[LABEL_COLUMN],
    )

    train = train.assign(**{SPLIT_COLUMN: TRAIN_SPLIT})
    validation = validation.assign(**{SPLIT_COLUMN: VALIDATION_SPLIT})
    test = test.assign(**{SPLIT_COLUMN: TEST_SPLIT})

    manifest = pd.concat(
        [train, validation, test],
        ignore_index=True,
    )

    manifest = manifest.sort_values(RECORD_ID_COLUMN).reset_index(drop=True)
    validate_manifest(manifest, expected_rows=len(cleaned))
    return manifest


def validate_manifest(manifest: pd.DataFrame, expected_rows: int) -> None:
    required = {RECORD_ID_COLUMN, LABEL_COLUMN, SPLIT_COLUMN}
    missing = required.difference(manifest.columns)

    if missing:
        raise DatasetPreparationError(
            f"Split manifest is missing columns: {', '.join(sorted(missing))}"
        )

    if len(manifest) != expected_rows:
        raise DatasetPreparationError(
            f"Manifest row count {len(manifest)} does not match {expected_rows}."
        )

    if manifest[RECORD_ID_COLUMN].duplicated().any():
        raise DatasetPreparationError("Manifest record identifiers are not unique.")

    observed_splits = set(manifest[SPLIT_COLUMN].unique())
    expected_splits = {TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT}

    if observed_splits != expected_splits:
        raise DatasetPreparationError(
            f"Expected splits {sorted(expected_splits)}, found {sorted(observed_splits)}."
        )

    for split_name, group in manifest.groupby(SPLIT_COLUMN):
        if set(group[LABEL_COLUMN].unique()) != {0, 1}:
            raise DatasetPreparationError(
                f"Split {split_name!r} does not contain both labels."
            )


def build_metadata(
    manifest: pd.DataFrame,
    preprocessing_summary: dict[str, int],
    source_md5: str,
    random_state: int = RANDOM_STATE,
) -> dict[str, object]:
    split_counts = {
        str(key): int(value)
        for key, value in manifest[SPLIT_COLUMN].value_counts().sort_index().items()
    }

    class_counts = {
        split_name: {
            str(label): int(count)
            for label, count in group[LABEL_COLUMN].value_counts().sort_index().items()
        }
        for split_name, group in manifest.groupby(SPLIT_COLUMN, sort=True)
    }

    return {
        "source_dataset": str(SOURCE_DATASET),
        "source_md5": source_md5,
        "random_state": random_state,
        "split_fractions": {
            TRAIN_SPLIT: TRAIN_FRACTION,
            VALIDATION_SPLIT: VALIDATION_FRACTION,
            TEST_SPLIT: TEST_FRACTION,
        },
        "preprocessing": preprocessing_summary,
        "manifest_rows": int(len(manifest)),
        "split_counts": split_counts,
        "class_counts": class_counts,
        "record_id_algorithm": "SHA-256 over the normalised title/body identity",
    }


def prepare_dataset() -> tuple[pd.DataFrame, dict[str, object]]:
    """Loads, validates, preprocesses, splits, and writes deterministic manifests."""

    if not SOURCE_DATASET.is_file():
        raise FileNotFoundError(f"Dataset not found: {SOURCE_DATASET}")

    actual_md5 = calculate_file_hash(SOURCE_DATASET, "md5")

    if actual_md5.casefold() != SOURCE_MD5:
        raise DatasetPreparationError(
            f"Source MD5 mismatch. Expected {SOURCE_MD5}, found {actual_md5}."
        )

    source = pd.read_csv(SOURCE_DATASET, encoding="utf-8", low_memory=False)
    cleaned, summary = preprocess_welfake(source, mode="combined")
    manifest = create_split_manifest(cleaned)
    metadata = build_metadata(manifest, summary.to_dict(), actual_md5)

    PROCESSED_DIRECTORY.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(
        MANIFEST_PATH,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    metadata["manifest_sha256"] = calculate_file_hash(MANIFEST_PATH, "sha256")
    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return manifest, metadata


def main() -> None:
    print("Preparing deterministic WELFake dataset splits...")
    manifest, metadata = prepare_dataset()
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Metadata: {METADATA_PATH}")
    print(f"Rows assigned: {len(manifest):,}")


if __name__ == "__main__":
    main()
