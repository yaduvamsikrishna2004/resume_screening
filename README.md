# Resume Screening Web Application

This project is a web-based Resume Screening tool that uses machine learning and natural language processing to help recruiters and job seekers automatically evaluate and rank resumes against a given job description.

## Features

- **Upload multiple resumes** in PDF or TXT format.
- **Enter a job description** to match resumes against.
- **Automatic extraction** of text from resumes (PDF/TXT).
- **ML-based prediction** of resume category/fit.
- **Similarity scoring** between each resume and the job description.
- **Combined ranking** based on model confidence and similarity.
- **Results table** with rank, filename, predicted category, confidence, and similarity.
- **View uploaded resumes** and their content.
- **MongoDB integration** for storing uploaded resumes.

## Tech Stack

- **Backend:** Python, Flask, scikit-learn, PyMuPDF, MongoDB
- **Frontend:** HTML, CSS, JavaScript (AJAX)
- **ML Model:** Trained and saved as `model.pkl` (e.g., Logistic Regression with TF-IDF)

## Setup Instructions

1. **Clone the repository**
   ```sh
   git clone https://github.com/yourname/resume_screening.git
   cd resume_screening
   ```

2. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```

3. **Start MongoDB**
   - Make sure MongoDB is running locally on `mongodb://localhost:27017/`.

4. **Train or place your ML model**
   - Place your trained `model.pkl` in the `webapp/` directory.
   - (Optional) Use `pipeline/train_model.py` to train a new model.

5. **Run the Flask app**
   ```sh
   cd webapp
   python app.py
   ```

6. **Open in browser**
   - Go to [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Project Structure

```
resume_screening/
│
├── pipeline/
│   └── train_model.py      # Script to train and save the ML model
│
├── webapp/
│   ├── app.py              # Main Flask application
│   ├── templates/
│   │   ├── index.html      # Main frontend template
│   │   └── ranked_resumes.html # Ranked resumes view
│   ├── static/
│   │   └── Cover.png       # Illustration image
│   └── uploads/            # (Optional) Uploaded files
│
├── requirements.txt
└── README.md
```

## Usage

1. **Upload resumes** and enter a job description.
2. **Submit** to get ranked results.
3. **View details** or content of each uploaded resume.

## Notes

- Only `.pdf` and `.txt` files are supported.
- Make sure MongoDB is running before starting the app.
- The ML model must be compatible with scikit-learn and accept text input.

## License

MIT License

---

**Developed by [C yadu vamsi krishna]**
