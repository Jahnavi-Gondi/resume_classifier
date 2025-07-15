# Resume Category Classifier 🧠📄

A machine learning project that classifies resumes into various job categories using NLP and Logistic Regression.

---

## 🚀 Features

- Text preprocessing & cleaning
- TF-IDF vectorization
- Logistic Regression training
- 99%+ Accuracy on test set
- Predict using raw text or `.txt` file
- CLI-based interface

---

## 📁 Folder Structure

resume_classifier/
├── data/
│ ├── raw_resume.csv
│ └── cleaned_resumes.csv
├── models/
│ └── resume_classifier.pkl
├── scripts/
│ ├── preprocessing.py
│ ├── train.py
│ └── predict.py
├── README.md
└── requirements.txt

yaml
Copy code

---

## 💻 How to Run Locally

> Make sure you have Python 3.7+ installed.

1. Clone the Repo
```bash
git clone https://github.com/Jahnavi-Gondi/resume_classifier.git
cd resume_classifier
2. Install Dependencies
bash
pip install -r requirements.txt
3. Preprocess the Data
bash
python scripts/preprocessing.py
4. Train the Model
bash
python scripts/train.py
5. Predict Resume Category
bash
python scripts/predict.py


## 🔮 Sample Prediction
<img width="1501" height="494" alt="Sample Prediction Output" src="https://github.com/user-attachments/assets/0ac8cf13-8e60-44ec-9d1b-10fec6429616" />


## 🧰 Tech Stack
Python

Pandas

Scikit-learn

TF-IDF Vectorizer

Joblib

 ## 📊 Accuracy
Accuracy: 99.48%

Precision: 0.99

Recall: 1.00

F1-score: 0.99

## 📂 Dataset Used
The dataset used for training was collected from open-source GitHub repositories and resume datasets publicly available via:

Kaggle

## 🚧 Future Plans
Deploy the model using Streamlit or Flask

Add a web-based UI for uploading resumes

Improve classification with deep learning (e.g., BERT)

Integrate PDF to text extraction using pdfminer or PyMuPDF

## 👩‍💻 Author
Jahnavi Gondi
