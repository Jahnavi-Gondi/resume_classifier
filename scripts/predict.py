import os
import joblib
import re

# === Load model and vectorizer ===
model_path = os.path.join("models", "resume_classifier.pkl")
model, vectorizer = joblib.load(model_path)

# === Resume Cleaning Function ===
def clean_resume(text):
    text = str(text)
    text = re.sub(r"Ã.+?[\s:]", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.lower()
    noise_keywords = [
        "skill details", "education details", "company details", "description -",
        "personal skills", "project name", "role:", "key responsibilities", "tools & technologies"
    ]
    for keyword in noise_keywords:
        text = text.replace(keyword, " ")
    return text.strip()

# === Prediction Logic ===
def predict_resume_category(resume_text):
    cleaned_text = clean_resume(resume_text)
    vectorized = vectorizer.transform([cleaned_text])
    predicted_category = model.predict(vectorized)[0]
    return predicted_category

# === User Input ===
if __name__ == "__main__":
    print("\n=== Resume Category Predictor ===")
    choice = input("Enter 1 to type/paste resume text OR 2 to load from .txt file: ")

    if choice == "1":
        print("\nPaste the resume text:")
        resume_text = input()
    elif choice == "2":
        file_path = input("Enter the full path of the .txt file: ").strip()
        if not os.path.exists(file_path):
            print("File not found!")
            exit()
        with open(file_path, "r", encoding="utf-8") as file:
            resume_text = file.read()
    else:
        print("Invalid choice.")
        exit()

    category = predict_resume_category(resume_text)
    print(f"\n🔍 Predicted Resume Category: **{category}**")
