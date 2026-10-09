# AI Career Copilot

An NLP and machine learning project that predicts a resume–job description match score using text features.

## Overview

AI Career Copilot takes a job description and a resume as input and predicts a match score between 0 and 100. It uses a machine learning model trained on English job descriptions and resumes.

**Important:** The predicted score estimates the matching score represented in the training dataset. It is not a hiring probability or a validated ATS score.

## Features

- Accepts a job description and resume as input.
- Converts text into numerical features using TF-IDF.
- Predicts a match score using Ridge Regression.
- Evaluates model performance using MAE, RMSE, and R².

## Tech Stack

- **Language:** Python
- **NLP:** TF-IDF Vectorization
- **Machine Learning:** Scikit-learn, Ridge Regression
- **Dataset:** `batuhanmtl/job_resume_fit`
- **Model Persistence:** Joblib

## Dataset

The project uses an English dataset containing job descriptions, resumes, and AI-generated match scores.

- Dataset: [Job Resume Fit](https://huggingface.co/datasets/batuhanmtl/job_resume_fit)
- Rows after removing duplicate job–resume pairs: 2,383
- Prediction target: `ai_match_score`

## Model Performance

The current baseline model was evaluated on a held-out test set.

| Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | 12.29 |
| Root Mean Squared Error (RMSE) | 15.11 |
| R² Score | 0.6267 |

These results measure how closely the model predicts the dataset's scores. They do not establish real-world hiring accuracy.

## Project Structure

```text
AI-Career-Copilot/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   ├── data/
│   │   └── load_dataset.py
│   ├── features/
│   └── models/
│       ├── train_model.py
│       └── predict.py
├── tests/
├── requirements.txt
└── README.md
```

## Setup

Clone the repository:

```bash
git clone https://github.com/amritas0412/AI-Career-Copilot.git
cd AI-Career-Copilot
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train the Model

```bash
python src/models/train_model.py
```

Training saves the model and TF-IDF vectorizer under `src/models/`.

## Predict a Resume Match Score

```bash
python src/models/predict.py
```

Enter the job description, type `END` on a new line, and then enter the resume. Type `END` on a new line again to get the predicted score.

## Future Improvements

- Improve model performance and evaluate generalization.
- Compare TF-IDF with semantic text representations.
- Add more robust validation using suitable evaluation datasets.

## Author

Developed as a machine learning portfolio project.
