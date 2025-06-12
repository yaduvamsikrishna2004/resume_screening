import pandas as pd
from pymongo import MongoClient
import pickle

# Connect to MongoDB and fetch resumes
client = MongoClient("mongodb://localhost:27017/")
db = client["resume_screening"]
collection = db["predictions"]

# Fetch resumes and categories
docs = list(collection.find({"resume": {"$exists": True, "$ne": ""}}))
if not docs:
    print("No resumes found in the database.")
    exit()

# Build DataFrame
df = pd.DataFrame([{"resume": doc["resume"], "category": doc.get("category", ""), "filename": doc.get("filename", "")} for doc in docs])

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Get the resume texts (as a list)
resume_texts = df["resume"].astype(str).tolist()

# Predict categories and get confidence scores
predictions = model.predict(resume_texts)
if hasattr(model, "predict_proba"):
    proba = model.predict_proba(resume_texts)
    confidences = proba.max(axis=1)
else:
    confidences = [1.0] * len(predictions)  # fallback if model doesn't support predict_proba

# Add predictions and confidence to DataFrame and sort
df["predicted_category"] = predictions
df["confidence"] = confidences
df_sorted = df.sort_values(by="confidence", ascending=False).reset_index(drop=True)
df_sorted["rank"] = df_sorted.index + 1  # Rank starts at 1

# Display top matches with rank
print(df_sorted[["rank", "filename", "predicted_category", "confidence"]].head(10))

client.close()