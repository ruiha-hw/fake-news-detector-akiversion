# Dataset Source Directory

This directory is retained for dataset download and source-data preparation.

The raw WELFake dataset is not committed to Git because the CSV is approximately 233.73 MB and exceeds GitHub's standard per-file limit.

Place downloaded source data in this directory when required by the dataset acquisition or preparation workflow.

The deployed Flask application uses the trained pipeline under `model_artifacts/final` and does not require the raw dataset for ordinary prediction.
