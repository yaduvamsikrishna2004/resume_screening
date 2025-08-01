import os
import pickle
import tempfile
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from datetime import datetime
import fitz  # PyMuPDF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Flask setup
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'txt', 'pdf'}

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client["resume_screening"]
uploads_col = db["uploads"]

# Load ML model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# File validation
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Text extraction
def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    elif ext == '.pdf':
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    return ""

# Similarity ranking
def rank_resumes(job_desc, resumes):
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([job_desc] + resumes)
    similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    return similarities

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    job_desc = request.form["job_description"]
    files = request.files.getlist("resumes")

    resume_texts = []
    resume_names = []

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            temp_path = os.path.join(tempfile.gettempdir(), filename)
            file.save(temp_path)

            text = extract_text(temp_path)
            if text.strip():
                resume_texts.append(text)
                resume_names.append(filename)

                # Log to MongoDB
                uploads_col.insert_one({
                    "filename": filename,
                    "text": text,
                    "uploaded_at": datetime.utcnow()
                })

    if not resume_texts:
        return jsonify({"error": "No valid resumes uploaded."}), 400

    # Predict
    predictions = model.predict(resume_texts)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(resume_texts)
        confidences = proba.max(axis=1)
    else:
        confidences = [1.0] * len(predictions)

    # Similarity to job description
    similarities = rank_resumes(job_desc, resume_texts)

    # Combine and sort
    results = list(zip(resume_names, predictions, confidences, similarities))
    results.sort(key=lambda x: x[2], reverse=True)

    combined_scores = [(conf + sim) / 2 for _, _, conf, sim in results]
    results = [res + (score,) for res, score in zip(results, combined_scores)]
    results.sort(key=lambda x: x[4], reverse=True)
    results_with_rank = [
        [i + 1, name, pred, float(conf), float(sim), float(score)]
        for i, (name, pred, conf, sim, score) in enumerate(results)
    ]

    return jsonify({"results": results_with_rank})

# Run app
if __name__ == '__main__':
    app.run(debug=True)

