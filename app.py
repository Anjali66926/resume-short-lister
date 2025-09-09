import os
import re
from collections import Counter
from math import sqrt
from flask import Flask, render_template, request, redirect, url_for, flash
import pdfplumber
import docx

# ---------- Setup ----------
# NOTE: In a production environment, you should use a more secure method
# for storing and handling uploaded files.
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "resume-ai-secret"  # Required for flash messages

# Minimal stopword list
STOPWORDS = {
    "a","an","and","are","as","at","be","by","for","from","has","he","in","is",
    "it","its","of","on","that","the","to","was","were","will","with","i","you",
    "your","we","they","this","those","these","our","their","or","if","but","so",
    "than","then","there","here","over","under","into","out","about","across"
}

# ---------- Helpers ----------
def allowed_file(filename: str) -> bool:
    """Checks if a file extension is in the allowed set."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_docx(path: str) -> str:
    """Extracts text from a DOCX file."""
    try:
        doc = docx.Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        print(f"Error reading DOCX file: {e}")
        return ""

def read_pdf(path: str) -> str:
    """Extracts text from a PDF file."""
    try:
        text = []
        with pdfplumber.open(path) as pdf:
            for p in pdf.pages:
                t = p.extract_text() or ""
                text.append(t)
        return "\n".join(text)
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        return ""

def extract_text(path: str) -> str:
    """Dispatches to the correct text extraction function based on file extension."""
    if path.lower().endswith(".pdf"):
        return read_pdf(path)
    if path.lower().endswith(".docx"):
        return read_docx(path)
    return ""

def tokenize(text: str) -> list[str]:
    """
    Tokenizes text by converting to lowercase, removing non-alphanumeric characters,
    and filtering out stopwords.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s+]", " ", text)
    tokens = [t for t in text.split() if t and t not in STOPWORDS]
    return tokens

def cosine_similarity(tokens_a: list[str], tokens_b: list[str]) -> float:
    """
    Calculates the cosine similarity between two lists of tokens.
    This implementation does not require external libraries like scikit-learn.
    """
    fa, fb = Counter(tokens_a), Counter(tokens_b)
    # Calculate the dot product
    dot_product = sum(fa[w] * fb[w] for w in set(fa) & set(fb))
    # Calculate the norms
    norm_a = sqrt(sum(v*v for v in fa.values()))
    norm_b = sqrt(sum(v*v for v in fb.values()))
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
        
    return dot_product / (norm_a * norm_b)

def matched_keywords(tokens_resume: list[str], tokens_jd: list[str]) -> list[str]:
    """Finds common keywords between the resume and job description tokens."""
    resume_set = set(tokens_resume)
    jd_set = set(tokens_jd)
    # Show only meaningful words (length >= 3) to avoid noise
    return sorted([w for w in (resume_set & jd_set) if len(w) >= 3])

# ---------- Routes ----------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        jd_text = request.form.get('job_description', '').strip()
        files = request.files.getlist('resumes')

        if not jd_text:
            flash("Please paste a job description.", "error")
            return redirect(url_for('index'))

        if not files or all(f.filename == "" for f in files):
            flash("Please choose one or more resumes (PDF/DOCX).", "error")
            return redirect(url_for('index'))

        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

        # Preprocess JD once
        jd_tokens = tokenize(jd_text)

        results = []
        for f in files:
            if not f or f.filename == "":
                continue
            if not allowed_file(f.filename):
                flash(f"Skipped (unsupported type): {f.filename}", "warn")
                continue

            save_path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
            f.save(save_path)

            raw = extract_text(save_path)
            res_tokens = tokenize(raw)

            sim = cosine_similarity(res_tokens, jd_tokens)
            percent = round(sim * 100, 2)

            results.append({
                "filename": f.filename,
                "percent": percent,
                "matches": matched_keywords(res_tokens, jd_tokens)[:25]  # Cap display
            })
            
            # Clean up the uploaded file after processing
            os.remove(save_path)

        # Sort results by best match first
        results.sort(key=lambda x: x["percent"], reverse=True)
        return render_template('results.html', results=results, jd=jd_text)

    return render_template('index.html')

if __name__ == '__main__':
    # Create the uploads folder if it doesn't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
