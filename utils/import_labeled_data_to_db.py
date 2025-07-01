import pandas as pd
from pymongo import MongoClient

try:
    # Load CSV
    df = pd.read_csv(r"C:\Users\yaduv\OneDrive\Desktop\UpdatedResumeDataSet.csv", encoding='utf-8')
    df.columns = df.columns.str.strip()
except Exception as e:
    print("Error reading CSV:", e)
    exit()

try:
    # Connect to MongoDB
    client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
    db = client["resume_screening"]
    collection = db["predictions"]

    # Clear old data
    collection.delete_many({})

    # Prepare documents
    docs = [
        {"resume": str(row['Resume']).strip(), "category": str(row['Category']).strip()}
        for _, row in df.iterrows()
        if str(row['Resume']).strip() and str(row['Category']).strip()
    ]

    # Insert documents
    if docs:
        collection.insert_many(docs)
        print("Sample docs:", docs[:3])
    else:
        print("No valid documents to insert.")
except Exception as e:
    print("Error during MongoDB operations:", e)
finally:
    client.close()
    print("Import complete.")