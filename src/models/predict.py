
import joblib

model = joblib.load("src/models/resume_score_model.pkl")
vectorizer = joblib.load("src/models/tfidf_vectorizer.pkl")


def predict_resume_score(job_description, resume):
    input_text = job_description + " " + resume
    input_features = vectorizer.transform([input_text])

    predicted_score = model.predict(input_features)[0]
    predicted_score = max(0, min(100, predicted_score))

    return round(float(predicted_score), 2)


if __name__ == "__main__":
    print("Enter job description (type END on a new line when finished):")

    job_lines = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        job_lines.append(line)

    job_description = "\n".join(job_lines)

    print("\nEnter resume (type END on a new line when finished):")

    resume_lines = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        resume_lines.append(line)

    resume = "\n".join(resume_lines)

    score = predict_resume_score(job_description, resume)
    print(f"\nPredicted Resume Score: {score}/100")
