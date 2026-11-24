# audit_dataset.py
# Audits WELFake without modifying the verified source CSV.

from pathlib import Path
import hashlib
import json
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DATASET = (
    ROOT
    / "data"
    / "raw"
    / "WELFake_Dataset.csv"
)

REPORT_DIR = ROOT.parent / "documentation"

EXPECTED_MD5 = (
    "73c9675a4b3d09f86a6933d0b8d7d908"
)

SPACE = re.compile(r"\s+")


def md5(path):
    digest = hashlib.md5()

    with path.open("rb") as source:
        for block in iter(
            lambda: source.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def normalise(series):
    return (
        series
        .fillna("")
        .astype(str)
        .map(
            lambda value: SPACE.sub(
                " ",
                value,
            )
            .strip()
            .casefold()
        )
    )


def lengths(series):
    values = (
        series
        .fillna("")
        .astype(str)
        .str.len()
    )

    return {
        "minimum": int(values.min()),
        "maximum": int(values.max()),
        "mean": round(
            float(values.mean()),
            2,
        ),
        "median": round(
            float(values.median()),
            2,
        ),
        "p90": round(
            float(values.quantile(0.90)),
            2,
        ),
        "p95": round(
            float(values.quantile(0.95)),
            2,
        ),
        "p99": round(
            float(values.quantile(0.99)),
            2,
        ),
    }


def main():
    if not DATASET.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET}"
        )

    print(
        "Loading verified WELFake dataset "
        "in read-only mode..."
    )

    actual_md5 = md5(DATASET)

    frame = pd.read_csv(
        DATASET,
        encoding="utf-8",
        low_memory=False,
    )

    lookup = {
        str(column).strip().casefold(): column
        for column in frame.columns
    }

    required = {
        "title",
        "text",
        "label",
    }

    if not required.issubset(lookup):
        raise ValueError(
            "Required columns missing. "
            f"Found: {frame.columns.tolist()}"
        )

    title = lookup["title"]
    text = lookup["text"]
    label = lookup["label"]

    labels = pd.to_numeric(
        frame[label],
        errors="coerce",
    )

    normal_titles = normalise(
        frame[title]
    )

    normal_texts = normalise(
        frame[text]
    )

    pairs = (
        normal_titles
        + "\n"
        + normal_texts
    )

    conflicts = pd.DataFrame(
        {
            "pair": pairs,
            "label": labels,
        }
    )

    conflict_count = int(
        conflicts
        .groupby("pair")["label"]
        .nunique()
        .gt(1)
        .sum()
    )

    report = {
        "dataset": str(DATASET),
        "file_size_bytes": (
            DATASET.stat().st_size
        ),
        "expected_md5": EXPECTED_MD5,
        "actual_md5": actual_md5,
        "checksum_valid": (
            actual_md5.casefold()
            == EXPECTED_MD5
        ),
        "rows": int(len(frame)),
        "columns": [
            str(column)
            for column in frame.columns
        ],
        "data_types": {
            str(key): str(value)
            for key, value
            in frame.dtypes.items()
        },
        "missing_values": {
            str(key): int(value)
            for key, value
            in frame.isna().sum().items()
        },
        "empty_values": {
            str(title): int(
                normal_titles.eq("").sum()
            ),
            str(text): int(
                normal_texts.eq("").sum()
            ),
            str(label): int(
                labels.isna().sum()
            ),
        },
        "label_distribution": {
            str(key): int(value)
            for key, value
            in labels
            .value_counts(dropna=False)
            .sort_index()
            .items()
        },
        "duplicates": {
            "exact_rows": int(
                frame.duplicated().sum()
            ),
            "normalised_titles": int(
                normal_titles
                .duplicated(keep=False)
                .sum()
            ),
            "normalised_texts": int(
                normal_texts
                .duplicated(keep=False)
                .sum()
            ),
            "normalised_title_text_pairs": int(
                pairs
                .duplicated(keep=False)
                .sum()
            ),
            "conflicting_label_groups": (
                conflict_count
            ),
        },
        "length_statistics": {
            str(title): lengths(
                frame[title]
            ),
            str(text): lengths(
                frame[text]
            ),
        },
    }

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        REPORT_DIR
        / "WELFAKE_DATASET_AUDIT.json"
    )

    text_path = (
        REPORT_DIR
        / "WELFAKE_DATASET_AUDIT.txt"
    )

    formatted = json.dumps(
        report,
        indent=2,
        ensure_ascii=False,
    )

    json_path.write_text(
        formatted + "\n",
        encoding="utf-8",
    )

    text_path.write_text(
        formatted + "\n",
        encoding="utf-8",
    )

    print(formatted)

    print(
        f"Text report: {text_path}"
    )

    print(
        f"JSON report: {json_path}"
    )


if __name__ == "__main__":
    main()