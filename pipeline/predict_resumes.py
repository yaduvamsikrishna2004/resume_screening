import os
import pickle
from datetime import datetime, timezone
from pymongo import MongoClient

# Connect to MongoDB
def connect_to_mongo():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["resume_screening"]
    return db["predictions"]

# Load saved model
def load_model(model_path="model.pkl"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    with open(model_path, 'rb') as f:
        return pickle.load(f)

# Insert resume if not already present
def insert_resume_if_new(collection, filename, content):
    existing = collection.find_one({"filename": filename})
    if not existing:
        collection.insert_one({
            "filename": filename,
            "resume": content,
            "category": None,
            "timestamp": datetime.now(timezone.utc)
        })

# Predict resumes with no category
def predict_resumes(model_path="model.pkl", resume_folder="data/cleaned_resumes"):
    model = load_model(model_path)
    collection = connect_to_mongo()

    # Step 1: Insert resumes from folder if new
    for fname in os.listdir(resume_folder):
        if fname.endswith(".txt"):
            with open(os.path.join(resume_folder, fname), 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    insert_resume_if_new(collection, fname, content)

    # Step 2: Predict for documents with no category
    resumes = list(collection.find({"category": None, "resume": {"$exists": True}}))
    if not resumes:
        print("No new unlabeled resumes to predict.")
        return

    texts = [doc["resume"] for doc in resumes]
    predictions = model.predict(texts)

    # Step 3: Update predictions
    for doc, prediction in zip(resumes, predictions):
        collection.update_one(
            {"_id": doc["_id"]},
            {
                "$set": {
                    "category": prediction,
                    "predicted_at": datetime.now(timezone.utc)
                }
            }
        )
        print(f"{doc['filename']} → {prediction}")

    print("\n✅ All predictions completed and saved to MongoDB.")

# Entry point
if __name__ == "__main__":
    predict_resumes()
