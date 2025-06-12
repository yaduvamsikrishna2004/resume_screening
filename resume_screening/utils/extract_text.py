import os
import fitz  # PyMuPDF
import docx
import nltk
import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Setup
nltk.download("punkt")
nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))

def extract_text_from_pdf(file_path):
    text = ""
    try:
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
    return text

def extract_text_from_docx(file_path):
    text = ""
    try:
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        print(f"Error reading DOCX {file_path}: {e}")
    return text

def extract_text(file_path):
    if file_path.lower().endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.lower().endswith(".docx"):
        return extract_text_from_docx(file_path)
    else:
        print(f"Unsupported file type: {file_path}")
        return ""

def clean_text(text):
    try:
        if not text or not isinstance(text, str):
            return ""

        # Lowercase
        text = text.lower()

        # Tokenize
        tokens = word_tokenize(text)

        # Remove punctuation and stopwords
        tokens = [
            t for t in tokens
            if t.isalpha() and t not in STOPWORDS
        ]

        return ' '.join(tokens)

    except Exception as e:
        print(f"Error cleaning text: {e}")
        return ""


def extract_all_from_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".pdf", ".docx")):
            print(f"Processing: {filename}")
            file_path = os.path.join(input_folder, filename)
            raw_text = extract_text(file_path)
            cleaned = clean_text(raw_text)

            output_file = os.path.join(output_folder, filename.rsplit('.', 1)[0] + '.txt')
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(cleaned)

            print(f"Saved cleaned text to: {output_file}")

if __name__ == "__main__":
    input_dir = "data/resumes"
    output_dir = "data/cleaned_resumes"
    extract_all_from_folder(input_dir, output_dir)
