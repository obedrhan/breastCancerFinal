import os
import sqlite3
import re
import uuid
from datetime import datetime
from functools import wraps

from flask import (
    Flask, render_template, session, request,
    redirect, url_for, flash, send_from_directory
)
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

from skimage.io import imsave
from segmentation_feature import (
    load_dicom_as_image, contrast_enhancement, region_growing,
    morphological_operations, contour_extraction, crop_image_with_contours
)
from new_predict import predict_mammogram  # <- your ensemble predictor

app = Flask(__name__)
app.secret_key = 'your_secret_key'

UPLOAD_FOLDER = os.path.join(
    os.path.dirname(__file__), "static", "uploads"
)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'danger')
            return redirect('/')
        return f(*args, **kwargs)
    return decorated_function


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        fullname = request.form['fullname'].strip()
        email    = request.form['email'].strip()
        password = request.form['password'].strip()

        if not fullname or not email or not password:
            flash('All fields are required.', 'danger')
            return render_template('register.html')

        pw_regex = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)[A-Za-z\d]{8,}$'
        if not re.match(pw_regex, password):
            flash('Password must be 8+ chars with upper, lower & number.', 'danger')
            return render_template('register.html')

        conn = sqlite3.connect('radiologist_system.db')
        c = conn.cursor()
        c.execute('SELECT 1 FROM Radiologists WHERE email=?', (email,))
        if c.fetchone():
            flash('Email already registered.', 'danger')
            conn.close()
            return render_template('register.html')

        hashed = generate_password_hash(password, method='pbkdf2:sha256')
        c.execute(
            'INSERT INTO Radiologists (fullname,email,password) VALUES (?,?,?)',
            (fullname, email, hashed)
        )
        conn.commit(); conn.close()
        flash('Registered! Please log in.', 'success')
        return redirect('/')

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email    = request.form['email'].strip()
        password = request.form['password'].strip()

        if not email or not password:
            flash('Email & Password required.', 'danger')
            return render_template('login.html')

        conn = sqlite3.connect('radiologist_system.db')
        c = conn.cursor()
        c.execute('SELECT id,password FROM Radiologists WHERE email=?', (email,))
        row = c.fetchone()
        conn.close()

        if not row or not check_password_hash(row[1], password):
            flash('Invalid credentials.', 'danger')
            return render_template('login.html')

        session['user_id'] = row[0]
        flash('Logged in!', 'success')
        return redirect('/home')

    return render_template('login.html')


@app.route('/')
def index():
    session.clear()
    return render_template('index.html')


@app.route('/home')
@login_required
def home():
    return render_template('home.html')


@app.route('/view_previous', methods=['GET', 'POST'])
@login_required
def view_previous():
    radiologist_id = session['user_id']
    patient_name   = request.form.get('patient_name', '').strip()

    conn = sqlite3.connect('radiologist_system.db')
    c = conn.cursor()
    if patient_name:
        c.execute('''
            SELECT CR.classification_date, CR.prediction, CR.confidence, P.fullname
            FROM ClassificationResults CR
            JOIN Patients P ON CR.national_id=P.national_id
            WHERE CR.radiologist_id=? AND P.fullname LIKE ?
            ORDER BY CR.classification_date DESC
        ''', (radiologist_id, f"%{patient_name}%"))
    else:
        c.execute('''
            SELECT CR.classification_date, CR.prediction, CR.confidence, P.fullname
            FROM ClassificationResults CR
            JOIN Patients P ON CR.national_id=P.national_id
            WHERE CR.radiologist_id=?
            ORDER BY CR.classification_date DESC
        ''', (radiologist_id,))
    results = c.fetchall()
    conn.close()

    return render_template('previous_results.html',
                           results=results,
                           patient_name=patient_name)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/process', methods=['POST'])
@login_required
def process():
    # 1) Validate form
    patient_name = request.form['patient_name'].strip()
    national_id  = request.form['national_id'].strip()
    if not patient_name or not national_id:
        flash('Patient name & National ID required.', 'danger')
        return redirect('/home')

    # 2) Validate upload
    if 'file' not in request.files:
        flash('No file uploaded.', 'danger')
        return redirect('/home')
    f = request.files['file']
    if f.filename == '':
        flash('No file selected.', 'danger')
        return redirect('/home')

    try:
        # 3) Save DICOM
        filename = secure_filename(f.filename)
        full_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(full_path)

        # 4) Segment → crop ROI
        orig      = load_dicom_as_image(full_path)
        enh       = contrast_enhancement(orig)
        seed      = (enh.shape[1]//2, enh.shape[0]//2)
        grown     = region_growing(enh, seed, threshold=15)
        refined   = morphological_operations(grown)
        _, ctrs   = contour_extraction(refined)
        cropped   = crop_image_with_contours(orig, ctrs)

        # 5) Save ROI preview
        seg_fn    = f"segmented_{uuid.uuid4().hex}.png"
        seg_path  = os.path.join(app.config['UPLOAD_FOLDER'], seg_fn)
        imsave(seg_path, cropped)

        # 6) Run ensemble predictor
        result = predict_mammogram(full_path, cropped)

        # 7) Persist patient & result
        conn = sqlite3.connect('radiologist_system.db')
        c = conn.cursor()
        c.execute('SELECT 1 FROM Patients WHERE national_id=?', (national_id,))
        if not c.fetchone():
            c.execute('''
                INSERT INTO Patients
                  (national_id, fullname, radiologist_id, dob)
                VALUES (?,?,?,?)
            ''', (national_id,
                  patient_name,
                  session['user_id'],
                  datetime.now().strftime('%Y-%m-%d')))
            conn.commit()

        c.execute('''
            INSERT INTO ClassificationResults
              (national_id, radiologist_id, prediction, confidence)
            VALUES (?,?,?,?)
        ''', (national_id,
              session['user_id'],
              result["prediction"],
              result["confidence"]*100))
        conn.commit(); conn.close()

        # 8) Render result
        return render_template('result.html',
                               segmented_image=url_for('uploaded_file', filename=seg_fn),
                               prediction=result["prediction"],
                               confidence=result["confidence"]*100)

    except Exception as e:
        print("🔥 Error in /process:", e)
        flash(f"Processing error: {e}", 'danger')
        return redirect('/home')


@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out.', 'info')
    return redirect('/')


if __name__ == '__main__':
    app.run(debug=True)