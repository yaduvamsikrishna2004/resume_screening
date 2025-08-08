import os
import pickle
import tempfile
from flask import Flask, request, render_template, jsonify, redirect, url_for, session, send_from_directory
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from datetime import datetime
import fitz  # PyMuPDF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from flask_mail import Mail, Message
from werkzeug.security import generate_password_hash, check_password_hash

# Flask setup
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set a strong secret key
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'txt', 'pdf'}

# --- NEW: Create upload folder if it doesn't exist ---
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Mail configuration (update with your mail server settings)
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'your_gmail@gmail.com'
app.config['MAIL_PASSWORD'] = 'your_app_password'
mail = Mail(app)

# Admin credentials (update or use env variables in production)
ADMIN_EMAIL = 'admin@example.com'
ADMIN_PASSWORD = generate_password_hash('adminpassword')

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client["resume_screening"]
uploads_col = db["uploads"]
users_collection = db["users"]
contact_collection = db["contacts"]

# Dummy email notification function (you can customize)
def send_login_email(email, password):
    # msg = Message('Login Notification', recipients=[email])
    # msg.body = f'You have successfully logged in.\n\nEmail: {email}'
    # mail.send(msg)
    pass

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

@app.route('/resume')
def resume():
    if 'user' not in session:
        return redirect(url_for('signin'))
    return render_template('resume.html')


# --- MODIFIED: Prediction route with download link generation ---
@app.route('/predict', methods=['POST'])
def predict():
    job_desc = request.form["job_description"]
    files = request.files.getlist("resumes")

    resume_texts = []
    resume_names = []
    # --- NEW: List to store download URLs ---
    download_urls = []

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # --- MODIFIED: Save file to the persistent UPLOAD_FOLDER ---
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            text = extract_text(file_path)
            if text.strip():
                resume_texts.append(text)
                resume_names.append(filename)
                # --- NEW: Generate a download URL and add it to the list ---
                download_urls.append(url_for('download_resume', filename=filename))

                uploads_col.insert_one({
                    "filename": filename,
                    "text": text,
                    "uploaded_at": datetime.utcnow()
                })

    if not resume_texts:
        return jsonify({"error": "No valid resumes uploaded."}), 400

    predictions = model.predict(resume_texts)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(resume_texts)
        confidences = proba.max(axis=1)
    else:
        confidences = [1.0] * len(predictions)

    similarities = rank_resumes(job_desc, resume_texts)

    # --- MODIFIED: Zip results including the new download_urls list ---
    results = list(zip(resume_names, predictions, confidences, similarities, download_urls))
    results.sort(key=lambda x: x[2], reverse=True) # Sort by confidence

    # --- MODIFIED: Unpack results correctly for scoring ---
    combined_scores = [(conf + sim) / 2 for _, _, conf, sim, _ in results]
    results = [res + (score,) for res, score in zip(results, combined_scores)]
    results.sort(key=lambda x: x[5], reverse=True) # Sort by combined score (now at index 5)

    # --- MODIFIED: Add the download URL to the final JSON response ---
    # New format: [Rank, Name, Prediction, Confidence, Similarity, Download_URL, Combined_Score]
    results_with_rank = [
        [i + 1, name, pred, float(conf), float(sim), url, float(score)]
        for i, (name, pred, conf, sim, url, score) in enumerate(results)
    ]

    return jsonify({"results": results_with_rank})

# --- NEW: Download route to serve files ---
@app.route('/download/<filename>')
def download_resume(filename):
    """Serves a file for download from the UPLOAD_FOLDER."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)


# Contact route
@app.route('/contactus', methods=['GET', 'POST'])
def contactus():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        contact_message = {'name': name, 'email': email, 'message': message}
        contact_collection.insert_one(contact_message)
        msg = Message('Contact Form Submission', recipients=[email])
        msg.body = 'Thank you for reaching out! We will get back to you soon.'
        mail.send(msg)
        return jsonify({'status': 'success', 'message': 'Message sent successfully!'})
    return render_template('contactus.html')

# Signin route
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
            if not user.get('logged_in', False):
                send_login_email(user['email'], password)
                users_collection.update_one({"email": email}, {"$set": {"logged_in": True}})
            if user.get('form_submitted', False):
                return redirect(url_for('resume'))
            else:
                return redirect(url_for('resume'))
        return "Invalid credentials. Try again."
    return render_template('signin.html')

# Signup route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        mobile = request.form['mobile']
        password = request.form['password']
        hashed_password = generate_password_hash(password)
        users_collection.insert_one({
            "name": name,
            "email": email,
            "mobile": mobile,
            "password": hashed_password,
            "logged_in": False,
            "form_submitted": False
        })
        return redirect(url_for('signin'))
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

# Run app
if __name__ == '__main__':
    app.run(debug=True)