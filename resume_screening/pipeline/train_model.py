# resume_screening/pipeline/train_model.py

import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pymongo import MongoClient

def load_labeled_resumes_from_db():
    client = MongoClient("mongodb://localhost:27017/")
    try:
        db = client["resume_screening"]
        collection = db["predictions"]

        texts, labels = [], []

        for doc in collection.find():
            text = doc.get("resume", "") or doc.get("text", "")
            label = doc.get("category", "") or doc.get("predicted_category", "")

            if text.strip() and label.strip():
                texts.append(text.strip())
                labels.append(label.strip())

        if not texts:
            raise ValueError("No valid labeled resume data found in the database.")

        return texts, labels
    finally:
        client.close()

def train_model(model_path="model.pkl"):
    try:
        texts, labels = load_labeled_resumes_from_db()
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

        # Ensure directory exists
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {model_path}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    train_model()
