from datasets import load_dataset
import pandas as pd

DATASET_NAME = "Divyanandh/resume-matching-dataset-v2"

dataset = load_dataset(DATASET_NAME)

train_data = dataset["train"]

# Convert to Pandas DataFrame
df = train_data.to_pandas()

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nScore statistics:")
print(df[["resume_score", "selfintro_score", "total_score"]].describe())

print("\nFirst 10 total scores:")
print(df["total_score"].head(10).tolist())

print("\nUnique total scores:")
print(df["total_score"].nunique())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nChecking score consistency:")

score_difference = (
    df["resume_score"]
    + df["selfintro_score"]
    - df["total_score"]
)

print("Rows where scores don't add up:", (score_difference != 0).sum())
print("Maximum difference:", score_difference.abs().max())

print("\nResume grade distribution:")
print(df["resume_grade"].value_counts())

print("\nSelf-introduction grade distribution:")
print(df["selfintro_grade"].value_counts())

print("\nMost common total scores:")
print(df["total_score"].value_counts().head(20))

print("\nAverage total score by resume grade:")
print(df.groupby("resume_grade")["total_score"].agg(["count", "mean", "median"]))

print("\nAverage total score by self-introduction grade:")
print(df.groupby("selfintro_grade")["total_score"].agg(["count", "mean", "median"]))

print("\nAverage total score by grade combination:")

grade_combinations = (
    df.groupby(["resume_grade", "selfintro_grade"])["total_score"]
    .agg(["count", "mean", "median"])
)

print(grade_combinations)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

plt.hist(df["total_score"], bins=20)

plt.title("Distribution of Total Scores")
plt.xlabel("Total Score")
plt.ylabel("Number of Applications")

plt.show()