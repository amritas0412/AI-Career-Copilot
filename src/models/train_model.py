
from datasets import load_dataset
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATASET_NAME = "batuhanmtl/job_resume_fit"


# 1. Load the English dataset
dataset = load_dataset(DATASET_NAME)
df = dataset["train"].to_pandas()


# 2. Keep required columns and remove duplicate pairs
df = df[["resume_text", "job_text", "ai_match_score"]].dropna()
df = df.drop_duplicates(subset=["resume_text", "job_text"])


# 3. Prepare input text and target
X_text = (
    "Job Description: " + df["job_text"].astype(str)
    + " Resume: " + df["resume_text"].astype(str)
)

y = df["ai_match_score"].astype(float)


# 4. Split into training and testing sets
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text,
    y,
    test_size=0.2,
    random_state=42
)


# 5. Convert text into TF-IDF features
vectorizer = TfidfVectorizer(
    max_features=20000,
    stop_words="english"
)

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)


# 6. Train the model
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)


# 7. Make predictions and evaluate
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("\n===== ENGLISH DATASET MODEL RESULTS =====")
print(f"Dataset rows after deduplication: {len(df)}")
print(f"Training examples: {len(X_train_text)}")
print(f"Testing examples: {len(X_test_text)}")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.4f}")


# 8. Show sample predictions
results = pd.DataFrame({
    "actual_score": y_test.values,
    "predicted_score": predictions
})

print("\n===== SAMPLE PREDICTIONS =====")
print(results.head(10).round(2))


# 9. Save the model and vectorizer
joblib.dump(model, "src/models/resume_score_model.pkl")
joblib.dump(vectorizer, "src/models/tfidf_vectorizer.pkl")

print("\nModel and vectorizer saved successfully!")
