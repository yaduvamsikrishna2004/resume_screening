import os
import pickle
from flask import Flask, request, render_template, jsonify, redirect, url_for, session, send_from_directory, abort
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from datetime import datetime , timezone
import fitz  # PyMuPDF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from flask_mail import Mail, Message
from werkzeug.security import generate_password_hash, check_password_hash
import uuid
import flask

# --- 1. SETUP ---

app = Flask(__name__)
# NOTE: use a strong secret from env in prod
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev_only_change_me')

# Always use an absolute path for uploads (single source of truth)
UPLOAD_FOLDER = os.environ.get(
    "UPLOAD_FOLDER",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'txt', 'pdf'}
# Optional: protect against huge uploads (e.g., 16 MB)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Detect Flask version to choose correct send_from_directory parameter
FLASK_VERSION = tuple(map(int, flask.__version__.split('.')[:2]))
USE_PATH_PARAM = FLASK_VERSION >= (3, 0)

# Mail configuration (update with your real mail server settings)
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME', 'your_gmail@gmail.com')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD', 'your_app_password')
mail = Mail(app)

# Admin credentials (use environment variables in production)
ADMIN_EMAIL = os.environ.get('ADMIN_EMAIL', 'admin@example.com')
ADMIN_PASSWORD = generate_password_hash(os.environ.get('ADMIN_PASSWORD', 'adminpassword'))

# MongoDB setup
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/resume_screening")
client = MongoClient(MONGO_URI)
# If your URI includes a db name at the end, this picks it automatically; else fallback:
DEFAULT_DB = os.environ.get("MONGO_DB", "resume_screening")
db = client.get_database() if client.get_database().name else client[DEFAULT_DB]
uploads_col = db["uploads"]
users_collection = db["users"]
contact_collection = db["contacts"]

# Load ML model with error handling
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("\n--- WARNING: model.pkl not found. Prediction endpoint will not work. ---\n")
    model = None

# --- 2. HELPER FUNCTIONS ---

def allowed_file(filename):
    """Checks if a filename has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text(file_path):
    """Extracts text from a .txt or .pdf file."""
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    try:
        if ext == '.txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        elif ext == '.pdf':
            # Ensure the document is closed (avoids file-lock issues on Windows)
            with fitz.open(file_path) as doc:
                for page in doc:
                    text += page.get_text()
    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
    return text

def rank_resumes(job_desc, resumes):
    """Calculates cosine similarity between a job description and a list of resumes."""
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([job_desc] + resumes)
    similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    return similarities

# --- 3. CORE FLASK ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/resume')
def resume():
    if 'user' not in session:
        return redirect(url_for('signin'))
    model_loaded = model is not None
    return render_template('resume.html', model_loaded=model_loaded)

@app.route('/predict', methods=['POST'])
def predict():
    """Processes uploaded resumes, predicts categories, ranks them, and returns results."""
    if model is None:
        return jsonify({"error": "The prediction model is not loaded on the server. Please contact the administrator."}), 500

    job_desc = request.form.get("job_description", "")
    files = request.files.getlist("resumes")

    resume_data = []

    for file in files:
        if file and allowed_file(file.filename):
            # Secure once, then prepend a short UUID
            original_filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4().hex[:8]}_{original_filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)

            text = extract_text(file_path)
            if text.strip():
                resume_data.append({
                    "text": text,
                    "original_name": original_filename,
                    "unique_name": unique_filename
                })
                uploads_col.insert_one({
                    "original_filename": original_filename,
                    "stored_filename": unique_filename,
                    "text": text,
                    "uploaded_at": datetime.utcnow()
                })

    if not resume_data:
        return jsonify({"error": "No valid resumes were uploaded or text could not be extracted."}), 400

    resume_texts = [r['text'] for r in resume_data]
    predictions = model.predict(resume_texts)
    confidences = model.predict_proba(resume_texts).max(axis=1) if hasattr(model, "predict_proba") else [1.0] * len(predictions)
    similarities = rank_resumes(job_desc, resume_texts)

    results = []
    for i, data in enumerate(resume_data):
        combined_score = (confidences[i] + similarities[i]) / 2
        download_url = url_for('download_resume', filename=data['unique_name'])
        results.append({
            "name": data['original_name'],
            "prediction": predictions[i],
            "confidence": float(confidences[i]),
            "similarity": float(similarities[i]),
            "score": float(combined_score),
            "download_url": download_url
        })

    results.sort(key=lambda x: x['score'], reverse=True)
    ranked_results = [{**res, "rank": i + 1} for i, res in enumerate(results)]
    return jsonify({"results": ranked_results})

# --- Download route (Flask 2.x & 3.x compatible) ---
@app.route('/download/<filename>')
def download_resume(filename):
    """Serves a file for download from the UPLOAD_FOLDER."""
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Check if the file exists before sending
    if not os.path.isfile(file_path):
        abort(404, description=f"Resume '{filename}' not found.")

    if USE_PATH_PARAM:
        # Flask 3.x+ uses 'path'
        return send_from_directory(
            directory=app.config['UPLOAD_FOLDER'],
            path=filename,
            as_attachment=True
        )
    else:
        # Flask 2.x uses 'filename'
        return send_from_directory(
            directory=app.config['UPLOAD_FOLDER'],
            filename=filename,
            as_attachment=True
        )

# --- 4. USER AND CONTACT ROUTES ---

@app.route('/contactus', methods=['GET', 'POST'])
def contactus():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        contact_message = {'name': name, 'email': email, 'message': message, 'sent_at': datetime.now(timezone.utc)}
        contact_collection.insert_one(contact_message)
        try:
            msg = Message('Contact Form Submission', recipients=[email])
            msg.body = 'Thank you for reaching out! We will get back to you soon.'
            mail.send(msg)
        except Exception as e:
            print(f"Mail sending failed: {e}")  # Log error but don't block
        return jsonify({'status': 'success', 'message': 'Message sent successfully!'})
    return render_template('contactus.html')

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        if email == ADMIN_EMAIL and check_password_hash(ADMIN_PASSWORD, password):
            session['user'] = email
            return redirect(url_for('resume'))

        user = users_collection.find_one({"email": email})
        if user and check_password_hash(user['password'], password):
            session['user'] = user['email']
            return redirect(url_for('resume'))

        return "Invalid credentials. Please try again."
    return render_template('signin.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        mobile = request.form['mobile']
        password = request.form['password']

        if users_collection.find_one({"email": email}):
            return "An account with this email already exists.", 400

        users_collection.insert_one({
            "name": name,
            "email": email,
            "mobile": mobile,
            "password": generate_password_hash(password)
        })
        return redirect(url_for('signin'))
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))




# --- 5. RUN APP ---

if __name__ == '__main__':
    
    app.run(
        host="0.0.0.0",   # allows external access (important for deployment)
        port=5000,
        debug=False,      # turn off debug for production
        threaded=True     # multiple users at the same time
    )