import os
from datetime import datetime, timezone
from pymongo import MongoClient

# MongoDB connection
def connect_to_mongo():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["resume_screening"]
    return db["predictions"]

# Extract label from filename (e.g., "resume_john_data.txt" → "data")
def extract_label(filename):
    return filename.rsplit('.', 1)[0].split('_')[-1]

# Import all cleaned resumes from the folder into MongoDB
def import_cleaned_resumes(folder_path):
    collection = connect_to_mongo()

    for fname in os.listdir(folder_path):
        if fname.endswith(".txt"):
            path = os.path.join(folder_path, fname)

            with open(path, 'r', encoding='utf-8') as f:
                text = f.read().strip()

            if not text:
                print(f"Skipped empty file: {fname}")
                continue

            label = extract_label(fname)
            doc = {
                "filename": fname,
                "resume": text,
                "category": label,
                "timestamp": datetime.now(timezone.utc)
            }

            collection.insert_one(doc)
            print(f"Imported: {fname} -> {label}")

if __name__ == "__main__":
    import_cleaned_resumes("data/cleaned_resumes")
