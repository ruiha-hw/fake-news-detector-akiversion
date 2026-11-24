# WELFake Dataset and Data Processing

This document explains where the project data comes from, where the raw file must be placed, and how the data is processed before it reaches the TF-IDF and Logistic Regression pipeline.

## Dataset Source

The project uses the WELFake dataset.

Download the dataset from:

[Download the WELFake dataset from Kaggle](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification/download?datasetVersionNumber=73)

The published dataset contains news headlines, article text and binary labels.

The source label mapping is:

```text
0 = fake-labelled article
1 = real-labelled article
```

The raw CSV is approximately 233.73 MB, so it is not stored directly in this Git repository.

## Required Dataset Location

The downloaded archive can be placed and extracted inside:

```text
.dataset_source/
```

The required CSV must then be copied to:

```text
data/raw/WELFake_Dataset.csv
```

Expected structure:

```text
SD205FakeNewsAssesment/
â”œâ”€â”€ .dataset_source/
â”‚   â”œâ”€â”€ .gitignore
â”‚   â””â”€â”€ README.md
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ raw/
â”‚   â”‚   â”œâ”€â”€ README.md
â”‚   â”‚   â””â”€â”€ WELFake_Dataset.csv
â”‚   â””â”€â”€ processed/
â”‚       â””â”€â”€ split_manifest.csv
â”œâ”€â”€ model/
â”œâ”€â”€ model_artifacts/
â””â”€â”€ app.py
```

The `.dataset_source` directory is retained in Git through its own README and ignore policy. Downloaded archives and dataset files placed inside it remain local and are not committed accidentally.

## Raw Dataset Fields

The pipeline expects the source CSV to provide the following information:

```text
Title
Text
Label
```

The source dataset may also contain a serial or index column. That column is not used as model evidence because it does not describe the article.

## Data Processing Overview

The data-processing path is:

```text
WELFake_Dataset.csv
    â†“
Source-frame validation
    â†“
Whitespace normalisation
    â†“
Label validation
    â†“
Empty-record handling
    â†“
Normalised title-and-article identity
    â†“
Duplicate removal and conflict detection
    â†“
Deterministic SHA-256 record identity
    â†“
Locked train / validation / test assignment
    â†“
TF-IDF feature construction
    â†“
Logistic Regression
```

## 1. Source Validation

The preprocessing stage first confirms that the expected source fields are available.

The source data must contain:

```text
A title column
An article-text column
A label column
```

Labels must be numeric and binary:

```text
0
1
```

Records carrying unsupported, missing or conflicting labels are rejected rather than silently converted into another class.

## 2. Text Normalisation

Headline and article text are normalised before identity creation or feature construction.

Normalisation includes:

- handling missing text values;
- collapsing repeated whitespace;
- removing unnecessary leading and trailing spacing;
- preserving the written text used by the classifier;
- keeping title and article content as separate logical fields.

Case is preserved in the prepared records, while duplicate identity comparison is designed to remain stable against superficial case and whitespace differences.

## 3. Empty-Record Handling

A record can remain usable when only one text field contains information.

The processing rules allow:

```text
Headline with no article body
Article body with no headline
Headline and article body together
```

A record is removed when both the normalised headline and normalised article body are empty.

## 4. Duplicate Handling

Duplicate treatment is applied to the normalised headline-and-article pair.

This is important because the same article appearing in more than one data split could make model performance appear stronger than it really is.

The processing stage:

- creates a normalised identity key from the title and article text;
- removes repeated copies of the same labelled content;
- preserves the first valid occurrence;
- rejects conflicting duplicates where the same normalised content appears with different labels.

The raw source provided 72,134 accessible records. After validation, empty-record handling and normalised duplicate removal, 63,672 unique usable records remained.

## 5. Deterministic Record Identity

Each prepared record receives a deterministic SHA-256 record identifier derived from its normalised content.

The purpose of this identifier is to ensure that:

- the same prepared article receives the same identity;
- row-order changes do not redefine the record;
- duplicate record IDs can be detected;
- split assignments can be stored and checked independently of the source CSV order.

## 6. Locked Dataset Split

The prepared records are divided into three roles:

```text
Training:    70% - 44,570 records
Validation:  15% - 9,551 records
Test:        15% - 9,551 records
```

The split is stratified so both labels remain represented across each section.

The split assignments are written to:

```text
data/processed/split_manifest.csv
```

The manifest records the stable record identity and its assigned split.

The split boundary is deliberate:

- training data fits candidate pipelines;
- validation data compares feature methods and later supports confidence-threshold auditing;
- held-out test data measures the selected configuration once after model selection.

The held-out test split is not used to select the feature mode or interpretation threshold.

## 7. Model Input Methods

Three feature methods were compared.

### Headline only

Uses only the normalised title.

```text
Validation macro F1: 90.77%
```

### Article text only

Uses only the normalised article body.

```text
Validation macro F1: 95.34%
```

### Combined headline and article

Uses both fields in a locked structure:

```text
TITLE:
<headline>

ARTICLE:
<article body>
```

```text
Validation macro F1: 96.24%
```

The combined method produced the strongest validation result and became the selected runtime feature mode.

## 8. TF-IDF Feature Construction

The selected text is converted into numerical features with TF-IDF.

The final vectoriser uses:

```text
Maximum features:            100,000
Word range:                  unigrams and bigrams
Minimum document frequency:  2
Maximum document frequency:  98%
Sublinear term frequency:    enabled
```

Unigrams represent individual words.

Bigrams represent local two-word combinations.

TF-IDF increases the influence of terms that are useful inside a document while reducing the influence of terms that appear too commonly across the full prepared dataset.

## 9. Logistic Regression

The TF-IDF feature matrix is passed into Logistic Regression.

The final classifier uses:

```text
Solver:              liblinear
Regularisation C:    1.0
Maximum iterations:  1,000
Classes:             0 and 1
```

The classifier returns probabilities for both learned classes.

The larger probability becomes the raw model preference, while both probabilities remain available to the Flask runtime and frontend.

## 10. Final Evaluation

The selected combined configuration was evaluated against the untouched held-out test split.

```text
Held-out test records:  9,551
Held-out accuracy:      96.42%
Held-out macro F1:      96.39%
```

After final evaluation, the selected configuration was refitted using the training and validation records and saved as the deployable pipeline.

The final runtime model is stored at:

```text
model_artifacts/final/fake_news_pipeline.joblib
```

The associated metadata and final metrics are stored at:

```text
model_artifacts/final/model_metadata.json
model_artifacts/final/final_test_metrics.json
```

## 11. Runtime Dataset Requirement

The raw dataset is required for:

- preprocessing reproduction;
- split generation;
- model experimentation;
- model retraining;
- held-out evaluation reproduction.

The raw dataset is not required for ordinary browser prediction.

The Flask application loads the saved pipeline from:

```text
model_artifacts/final/fake_news_pipeline.joblib
```

This means a user can install and run the completed MVP without downloading the 233.73 MB source CSV, provided the saved final model artefacts are present.

## 12. Data Integrity Boundaries

The project uses several evidence controls around the data and model lifecycle:

- deterministic normalisation;
- conflicting-label rejection;
- duplicate removal before splitting;
- SHA-256 record identities;
- a locked split manifest;
- validation-only model selection;
- held-out final evaluation;
- final model SHA-256 verification at runtime.

These controls reduce the risk of accidental data leakage, stale model loading and unrepeatable evaluation results.

## Important Limitation

The labels describe the source dataset classes.

The model learns language patterns associated with those labels. It does not independently establish whether a submitted claim is factually true or false.

High confidence means the submitted language strongly resembles one learned class. It does not mean that the system has verified external evidence.

