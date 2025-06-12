# Resume Screening AI Web App

**Author:** Yadu Vamsi Krishna

---

## Overview

This project is an AI-powered resume screening system. It predicts the category of each resume and ranks candidates based on their relevance to a given job description. The system uses machine learning (Logistic Regression + TF-IDF), MongoDB for storage, and a Flask web interface.

---

## Features

- Upload `.txt` or `.pdf` resumes via a web interface.
- Predicts resume category using a trained ML model.
- Ranks resumes by combining model confidence and similarity to a job description.
- Stores resumes and predictions in MongoDB.
- Retrain model from labeled data in MongoDB.

---

## Project Structure

```
resume_screening/
│
├── pipeline/
│   └── train_model.py         # Script to train and save the ML model
│
├── utils/
│   ├── preprocesser.py        # Preprocessing and helper functions
│   └── rank_resumes.py        # Script to rank resumes from MongoDB
│
├── webapp/
│   ├── app.py                 # Flask web application
│   └── templates/
│       ├── screening.html     # Upload form
│       └── results.html       # Results display
│
├── data/
│   ├── labeled_resumes/       # Folder for labeled training resumes
│   └── cleaned_resumes/       # Folder for resumes to be screened
│
├── model.pkl                  # Trained ML model (generated after training)
└── README.md                  # Project documentation
```

---

## Setup Instructions

### 1. Clone the repository
```sh
git clone <your-repo-url>
cd resume_screening
```

### 2. Install dependencies
```sh
pip install -r requirements.txt
```
*(Make sure you have MongoDB running locally on `localhost:27017`.)*

### 3. Prepare labeled data
- Place labeled resumes in `data/labeled_resumes/` (filename format: `resume_name_category.txt`).

### 4. Train the model
```sh
python pipeline/train_model.py
```
- This will train a model from labeled resumes in MongoDB and save it as `model.pkl`.

### 5. Run the web app
```sh
python webapp/app.py
```
- Open your browser at [http://localhost:5000](http://localhost:5000).

---

## Usage

1. Go to the web app and upload one or more resumes (`.txt` or `.pdf`).
2. Enter a job description in the provided field.
3. Submit to see:
    - Predicted category for each resume.
    - Model confidence.
    - Similarity to the job description.
    - Combined score and rank.

---

## How It Works

- **Model Training:**  
  `train_model.py` loads labeled resumes from MongoDB, trains a TF-IDF + Logistic Regression pipeline, and saves it as `model.pkl`.

- **Prediction & Ranking:**  
  The web app loads `model.pkl` and, for each uploaded resume:
    - Predicts the category and confidence.
    - Calculates cosine similarity to the job description.
    - Combines confidence and similarity for ranking.

- **Database:**  
  All uploaded resumes and predictions are stored in MongoDB for future reference and retraining.

---

## Notes

- Make sure MongoDB is running before using the app.
- You can retrain the model anytime by running `python pipeline/train_model.py`.
- For best results, use clear and labeled training data.

---

## License

MIT License

---

## Author

Yadu Vamsi Krishna
