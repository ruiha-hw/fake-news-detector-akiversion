<<<<<<< HEAD
# fake-news-detector aki version description:
A machine learning web application for fake news detection using TF-IDF and Logistic Regression.

# Overview:
This project aims to develop a machine learning web application that is capable of being able to identify whether news contents are likely real or fake.

# Techology stack:
- Python
- Flask
- Scikit-Learn
- TF-IDF
- Logistic Regression
- HTML
- CSS
- Bootstrap
- GitHub

# Team Roles:
- ML/Data: Victor
Dataset preparation, preprocessing, model training and evaluation.

- UI/UX: Aki
Wireframes, style guides, interface design and usability testing.

- Backend: James
Application integration, Flask development and model deployment.

# Project structure:
(TBA)
=======
# SD205 Fake News Assessment

Fake News Detection MVP developed for SD205 Assessment 2.

This repository contains the completed end-to-end application together with a filtered branch showing my individual contribution.

## Repository Structure

The repository uses two branches to separate the completed product from individual contribution evidence.

### `main`

The `main` branch contains the completed end-to-end application:

- WELFake data preparation
- TF-IDF feature extraction
- Logistic Regression classification
- Final trained ML pipeline
- Flask application and API
- Model-service integration
- Integrated frontend
- Confidence-threshold audit
- Three-state interpretation policy
- Automated tests
- Project documentation

### `james-integration`

The filtered branch contains the work attributable to my specialist contribution:

- Flask application factory
- Backend-to-frontend integration
- Model-service boundary
- Model SHA-256 verification
- Health and prediction endpoints
- Request validation
- Structured API responses
- Integrated testing frontend
- CSS and JavaScript
- Confidence-threshold audit
- Runtime interpretation policy
- Integration and frontend tests

The machine-learning development direction, including WELFake, TF-IDF and Logistic Regression, represents Victor's specialist contribution.

My responsibility was integrating the exported machine-learning pipeline into a controlled Flask runtime and creating the frontend used to operate, test and demonstrate the complete application.

## Project Objective

The application accepts:

- a news headline;
- article text;
- or both together.

The submitted text follows this runtime path:

```text
User input
    ↓
Flask validation
    ↓
Combined headline and article formatting
    ↓
TF-IDF vectorisation
    ↓
Logistic Regression
    ↓
Fake-pattern and real-pattern probabilities
    ↓
75% interpretation policy
    ↓
Frontend result
```

The application is a text-pattern classifier rather than an external fact-checking service.

It identifies whether submitted text more strongly resembles fake-labelled or real-labelled articles from the training dataset. It does not independently prove whether a claim is true or false.

## Main Results

```text
Prepared unique records:       63,672
Held-out test records:          9,551
Held-out accuracy:              96.42%
Held-out macro F1:              96.39%
Interpretation threshold:       75%
Decisive validation accuracy:   99.16%
Automated tests:                145 passed
```

## Result Interpretation

The raw model remains a binary classifier:

```text
0 = fake-labelled pattern
1 = real-labelled pattern
```

The application applies a separate user-facing interpretation policy.

### Confidence below 75%

```text
SEEK FURTHER INFORMATION
```

The model has a preferred class, but the probability does not clear the validation-backed threshold required for a decisive result.

### Fake-pattern probability of at least 75%

```text
POTENTIALLY MISLEADING
```

The submitted text strongly resembles fake-labelled training patterns.

### Real-pattern probability of at least 75%

```text
MORE CONSISTENT WITH REAL-NEWS PATTERNS
```

The submitted text strongly resembles real-labelled training patterns. This does not establish that its claims are factually true.

## Production-Quality Controls

The implementation includes:

- deterministic text preprocessing;
- normalised duplicate handling;
- stable SHA-256 record identities;
- locked training, validation and held-out test assignments;
- validation-based model selection;
- isolated held-out evaluation;
- final model SHA-256 verification;
- lazy model loading;
- application-factory separation;
- structured JSON responses;
- input and request-size validation;
- model-health reporting;
- confidence-threshold auditing;
- automated runtime and frontend testing.

The deployed model artefact is SHA-256 checked before loading. This prevents a stale, corrupted or replaced model file from silently entering the runtime under the expected filename.

## WELFake Dataset

The raw dataset is not stored in this repository because `WELFake_Dataset.csv` is approximately 233.73 MB and exceeds GitHub's standard file-size limit.

[Download the WELFake dataset from Kaggle](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification/download?datasetVersionNumber=73)

The source dataset provides the following main fields:

```text
Serial number
Title
Text
Label
```

The project uses this label mapping:

```text
0 = fake
1 = real
```

## Dataset Placement

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
├── .dataset_source/
├── data/
│   ├── README.md
│   ├── raw/
│   │   └── WELFake_Dataset.csv
│   └── processed/
│       └── split_manifest.csv
├── model/
├── model_artifacts/
├── application/
├── templates/
├── static/
├── tests/
└── app.py
```

Detailed processing instructions are available in [`data/README.md`](data/README.md).

The saved Flask runtime does not require the raw dataset for ordinary prediction. The raw data is required only when reproducing preprocessing, splitting, experimentation, training or held-out evaluation.

## Software Requirements

The verified project environment uses:

```text
Python 3.11.0
Flask
Pandas
NumPy
Scikit-learn
Joblib
Pytest
```

## Installation

Clone the repository:

```powershell
git clone https://github.com/DeusEx-Projects-Group/SD205FakeNewsAssesment.git
```

Enter the repository:

```powershell
Set-Location SD205FakeNewsAssesment
```

Create the virtual environment:

```powershell
py -3.11 -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Running the Test Suite

```powershell
python -m pytest tests -v
```

Expected final result:

```text
145 passed
```

## Running the Application

```powershell
python app.py
```

Open the application in a browser:

```text
http://127.0.0.1:5000
```

The status indicator should change from `Checking model` to `Model ready`.

Enter a headline, article text, or both, then select the classification action.

Stop the server with `Ctrl+C`.

## Public API

### Model health

```http
GET /health
```

Confirms that the saved model can be verified and loaded.

### Prediction

```http
POST /predict
Content-Type: application/json
```

Example request:

```json
{
  "title": "Example headline",
  "article": "Example article text"
}
```

The response includes:

- raw model label;
- original model classification;
- user-facing interpretation;
- confidence;
- fake-pattern probability;
- real-pattern probability;
- decisive status;
- interpretation threshold;
- user guidance;
- input character count.

## Important Limitation

This application measures learned language patterns.

It does not:

- search for external evidence;
- verify named sources;
- prove whether an event occurred;
- establish scientific plausibility;
- or replace human investigation.

A high model probability means that the submitted language strongly resembles one of the learned classes. It does not mean that the application has independently established truth.
>>>>>>> team-version
