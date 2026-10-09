from datasets import load_dataset
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATASET_NAME = "Divyanandh/resume-matching-dataset-v2"


# 1. Load dataset
dataset = load_dataset(DATASET_NAME)

train_df = dataset["train"].to_pandas()
test_df = dataset["test"].to_pandas()


# 2. Create input text
train_text = (
    train_df["jobpost"].fillna("")
    + " "
    + train_df["resume"].fillna("")
)

test_text = (
    test_df["jobpost"].fillna("")
    + " "
    + test_df["resume"].fillna("")
)


# 3. Target
y_train = train_df["resume_score"]
y_test = test_df["resume_score"]


# 4. Convert text into TF-IDF features
vectorizer = TfidfVectorizer(
    max_features=20000,
    stop_words=None
)

X_train = vectorizer.fit_transform(train_text)
X_test = vectorizer.transform(test_text)


# 5. Train ML model
model = Ridge(alpha=1.0)

model.fit(X_train, y_train)


# 6. Make predictions
predictions = model.predict(X_test)

# 9. Save the trained model and vectorizer
joblib.dump(model, "src/models/resume_score_model.pkl")
joblib.dump(vectorizer, "src/models/tfidf_vectorizer.pkl")

print("\nModel and vectorizer saved successfully!")


# 7. Evaluate
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)


print("\n===== MODEL RESULTS =====")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.4f}")


# 8. Show some predictions
results = pd.DataFrame({
    "actual_score": y_test.values,
    "predicted_score": predictions
})

print("\n===== SAMPLE PREDICTIONS =====")
print(results.head(10))