from pymongo import MongoClient
from collections import Counter

def connect_to_mongo():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["resume_screening"]
    return db["predictions"]

def check_label_distribution():
    collection = connect_to_mongo()
    resumes = collection.find({"category": {"$exists": True}})
    
    labels = [doc["category"].strip().lower() for doc in resumes if "category" in doc and doc["category"]]
    
    distribution = Counter(labels)
    
    print("=== Label Distribution ===")
    for label, count in distribution.items():
        print(f"{label}: {count}")

if __name__ == "__main__":
    check_label_distribution()
