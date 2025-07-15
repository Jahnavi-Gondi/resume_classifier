import pandas as pd
import re
import os

# Define paths relative to this script's location
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw_resume.csv')
CLEANED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'cleaned_resumes.csv')

# Load raw dataset
df = pd.read_csv(RAW_DATA_PATH)

# Drop nulls in key columns
df.dropna(subset=["Resume", "Category"], inplace=True)

def clean_resume(text):
    text = str(text)
    # Fix common encoding issues
    text = re.sub(r"Ã.+?[\s:]", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # Remove non-ASCII chars
    text = re.sub(r"\s{2,}", " ", text)         # Remove multiple spaces
    text = text.lower()
    
    # Remove common noisy keywords
    noise_keywords = [
        "skill details", "education details", "company details", "description -",
        "personal skills", "project name", "role:", "key responsibilities", "tools & technologies"
    ]
    
    for keyword in noise_keywords:
        text = text.replace(keyword, " ")
    
    return text.strip()

# Apply cleaning to resumes
df["Cleaned_Resume"] = df["Resume"].apply(clean_resume)

# Save cleaned dataset with correct path
df[["Cleaned_Resume", "Category"]].rename(columns={
    "Cleaned_Resume": "Resume"
}).to_csv(CLEANED_DATA_PATH, index=False)

print(f"Cleaned dataset saved as '{CLEANED_DATA_PATH}'")
