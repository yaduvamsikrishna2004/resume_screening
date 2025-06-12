import os
from datetime import datetime
from cachelib import MongoDbCache
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# MongoDB Setup
def connect_to_mongo():
    client = MongoClient("mongodb://localhost:27017/")  # or replace with Atlas URL
    db = client["resume_screening"]
    collection = db["predictions"]
    return collection

# Store prediction in MongoDB
def save_prediction_to_mongo(collection, filename, category):
    prediction_doc = {
        "filename": filename,
        "predicted_category": category,
        "timestamp": datetime.utcnow()
    }
    collection.insert_one(prediction_doc)

# Extract label from filename
def extract_label(filename):
    return filename.rsplit('.', 1)[0].split('_')[-1]

# Load labeled resumes
def load_labeled_resumes(folder):
    texts = []
    labels = []
    filenames = []
    for fname in os.listdir(folder):
        if fname.endswith(".txt"):
            path = os.path.join(folder, fname)
            with open(path, "r", encoding="utf-8") as f:
                texts.append(f.read())
                labels.append(extract_label(fname))
                filenames.append(fname)
    return texts, labels, filenames

# Train and evaluate model
def train_and_evaluate(folder):
    texts, labels, _ = load_labeled_resumes(folder)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.3, random_state=42
    )
    model = make_pipeline(
        TfidfVectorizer(stop_words='english'),
        LogisticRegression(max_iter=1000)
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    return model

# Predict a new resume
def predict_resume_category(model, resume_text):
    return model.predict([resume_text])[0]

# Main
if __name__ == "__main__":
    collection = connect_to_mongo()
    folder = "data/labeled_resumes"
    model = train_and_evaluate(folder)

    # Example resume prediction
    new_resumes = {
        "resume_john.txt": "Experienced in Java, Spring Boot, Docker, and CI/CD pipelines.",
        "resume_lisa.txt": "Built deep learning models using TensorFlow and PyTorch."
    }

    for fname, content in new_resumes.items():
        predicted = predict_resume_category(model, content)
        print(f"{fname} → {predicted}")
        save_prediction_to_mongo(collection, fname, predicted)
