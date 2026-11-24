# refine_dataset_audit.py
# Refines WELFake duplicate, usability, and extreme-length analysis.

from pathlib import Path
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

REPORT_DIRECTORY = (
    ROOT.parent
    / "documentation"
)

REPORT_PATH = (
    REPORT_DIRECTORY
    / "WELFAKE_REFINED_AUDIT.json"
)

SPACE = re.compile(r"\s+")


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


def main():
    print("Running refined WELFake audit...")

    frame = pd.read_csv(
        DATASET,
        encoding="utf-8",
        low_memory=False,
    )

    titles = normalise(frame["title"])
    texts = normalise(frame["text"])
    labels = pd.to_numeric(
        frame["label"],
        errors="coerce",
    )

    pairs = titles + "\n" + texts

    both_empty = (
        titles.eq("")
        & texts.eq("")
    )

    pair_frame = pd.DataFrame(
        {
            "pair": pairs,
            "label": labels,
        }
    )

    title_frame = pd.DataFrame(
        {
            "title": titles,
            "label": labels,
        }
    )

    text_frame = pd.DataFrame(
        {
            "text": texts,
            "label": labels,
        }
    )

    usable_mask = ~both_empty

    usable_pairs = pairs[usable_mask]

    pair_group_sizes = (
        usable_pairs
        .value_counts()
    )

    unique_pair_count = int(
        usable_pairs.nunique()
    )

    redundant_pair_rows = int(
        len(usable_pairs)
        - unique_pair_count
    )

    title_conflicts = int(
        title_frame[
            title_frame["title"].ne("")
        ]
        .groupby("title")["label"]
        .nunique()
        .gt(1)
        .sum()
    )

    text_conflicts = int(
        text_frame[
            text_frame["text"].ne("")
        ]
        .groupby("text")["label"]
        .nunique()
        .gt(1)
        .sum()
    )

    longest_indices = (
        frame["text"]
        .fillna("")
        .astype(str)
        .str.len()
        .nlargest(10)
        .index
    )

    longest_records = []

    for index in longest_indices:
        longest_records.append(
            {
                "row_index": int(index),
                "label": int(frame.at[index, "label"]),
                "title": str(
                    frame.at[index, "title"]
                )[:300],
                "text_length": int(
                    len(
                        str(
                            frame.at[index, "text"]
                        )
                    )
                ),
            }
        )

    report = {
        "source_rows": int(len(frame)),
        "both_title_and_text_empty": int(
            both_empty.sum()
        ),
        "rows_usable_before_deduplication": int(
            usable_mask.sum()
        ),
        "unique_normalised_title_text_pairs": (
            unique_pair_count
        ),
        "redundant_pair_rows": (
            redundant_pair_rows
        ),
        "largest_duplicate_group": int(
            pair_group_sizes.max()
        ),
        "duplicate_groups": int(
            pair_group_sizes.gt(1).sum()
        ),
        "title_groups_with_conflicting_labels": (
            title_conflicts
        ),
        "text_groups_with_conflicting_labels": (
            text_conflicts
        ),
        "pair_groups_with_conflicting_labels": int(
            pair_frame
            .groupby("pair")["label"]
            .nunique()
            .gt(1)
            .sum()
        ),
        "pair_group_size_distribution": {
            str(size): int(count)
            for size, count
            in pair_group_sizes
            .value_counts()
            .sort_index()
            .items()
        },
        "ten_longest_records": longest_records,
    }

    REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    formatted = json.dumps(
        report,
        indent=2,
        ensure_ascii=False,
    )

    REPORT_PATH.write_text(
        formatted + "\n",
        encoding="utf-8",
    )

    print(formatted)
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()