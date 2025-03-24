import os
import uuid
import sqlite3
import re
from datetime import datetime
from functools import wraps
from flask import Flask, render_template, session, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from predict import predict_mammogram
from skimage.io import imsave
from segmentation_feature import crop_image_with_contours, load_dicom_as_image, contrast_enhancement, region_growing, morphological_operations, contour_extraction

app = Flask(__name__)
app.secret_key = 'your_secret_key'

UPLOAD_FOLDER = '/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Login required decorator
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
        fullname = request.form.get('fullname', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()

        if not fullname or not email or not password:
            flash('All fields are required.', 'danger')
            return render_template('register.html')

        password_regex = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)[A-Za-z\d]{8,}$'
        if not re.match(password_regex, password):
            flash('Password must be 8+ characters long, including an uppercase, lowercase, and a number.', 'danger')
            return render_template('register.html')

        conn = sqlite3.connect('radiologist_system.db')
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM Radiologists WHERE email = ?', (email,))
        if cursor.fetchone():
            flash('This email is already registered.', 'danger')
            conn.close()
            return render_template('register.html')

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        cursor.execute('''INSERT INTO Radiologists (fullname, email, password) VALUES (?, ?, ?)''', (fullname, email, hashed_password))
        conn.commit()
        conn.close()
        flash('Registration successful! Please log in.', 'success')
        return redirect('/')

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()

        if not email or not password:
            flash('Email and Password are required.', 'danger')
            return render_template('login.html')

        conn = sqlite3.connect('radiologist_system.db')
        cursor = conn.cursor()
        cursor.execute('SELECT id, password FROM Radiologists WHERE email = ?', (email,))
        radiologist = cursor.fetchone()
        conn.close()

        if not radiologist or not check_password_hash(radiologist[1], password):
            flash('Invalid credentials.', 'danger')
            return render_template('login.html')

        session['user_id'] = radiologist[0]
        flash('Login successful!', 'success')
        return redirect('/home')

    return render_template('login.html')

@app.route('/home')
@login_required
def home():
    return render_template('home.html')

@app.route('/')
def index():
    session.clear()
    return render_template('index.html')

@app.route('/view_previous', methods=['GET', 'POST'])
@login_required
def view_previous():
    radiologist_id = session.get('user_id')

    conn = sqlite3.connect('radiologist_system.db')
    cursor = conn.cursor()
    patient_name = request.form.get('patient_name', '').strip() if request.method == 'POST' else ''

    if patient_name:
        cursor.execute('''
            SELECT CR.classification_date, CR.prediction, CR.confidence, P.fullname
            FROM ClassificationResults CR
            JOIN Patients P ON CR.national_id = P.national_id
            WHERE CR.radiologist_id = ? AND P.fullname LIKE ?
            ORDER BY CR.classification_date DESC
        ''', (radiologist_id, f"%{patient_name}%"))
    else:
        cursor.execute('''
            SELECT CR.classification_date, CR.prediction, CR.confidence, P.fullname
            FROM ClassificationResults CR
            JOIN Patients P ON CR.national_id = P.national_id
            WHERE CR.radiologist_id = ?
            ORDER BY CR.classification_date DESC
        ''', (radiologist_id,))

    results = cursor.fetchall()
    conn.close()
    return render_template('previous_results.html', results=results, patient_name=patient_name)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/process', methods=['POST'])
@login_required
def process():
    patient_name = request.form.get('patient_name', '').strip()
    national_id = request.form.get('national_id', '').strip()
    if not patient_name or not national_id:
        flash('Patient name and National ID are required.', 'danger')
        return redirect('/home')

    if 'file' not in request.files:
        flash('No file uploaded.', 'danger')
        return redirect('/home')

    file = request.files['file']
    if file.filename == '':
        flash('No file selected.', 'danger')
        return redirect('/home')

    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Load and segment the image
        original_image = load_dicom_as_image(file_path)
        enhanced = contrast_enhancement(original_image)
        seed_point = (enhanced.shape[1] // 2, enhanced.shape[0] // 2)
        grown = region_growing(enhanced, seed_point, threshold=15)
        refined = morphological_operations(grown)
        _, contours = contour_extraction(refined)
        cropped_img = crop_image_with_contours(original_image, contours)

        # Save segmented image
        segmented_filename = f"segmented_{os.path.splitext(filename)[0]}.png"
        segmented_path = os.path.join(app.config['UPLOAD_FOLDER'], segmented_filename)
        imsave(segmented_path, cropped_img)

        # Prediction
        result = predict_mammogram(file_path)

        conn = sqlite3.connect('radiologist_system.db')
        cursor = conn.cursor()
        radiologist_id = session.get('user_id')

        cursor.execute('SELECT national_id FROM Patients WHERE national_id = ?', (national_id,))
        patient = cursor.fetchone()
        if not patient:
            cursor.execute(
                'INSERT INTO Patients (national_id, fullname, radiologist_id, dob) VALUES (?, ?, ?, ?)',
                (national_id, patient_name, radiologist_id, datetime.now().strftime('%Y-%m-%d'))
            )
            conn.commit()

        cursor.execute(
            'INSERT INTO ClassificationResults (national_id, radiologist_id, prediction, confidence) VALUES (?, ?, ?, ?)',
            (national_id, radiologist_id, result["prediction"], result["confidence"])
        )
        conn.commit()
        conn.close()

        return render_template(
            'result.html',
            segmented_image=url_for('uploaded_file', filename=segmented_filename),
            prediction=result["prediction"],
            confidence=result["confidence"]
        )

    except Exception as e:
        print(f"🔥 Error: {e}")
        flash(f"Error during processing: {str(e)}", 'danger')
        return redirect('/home')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out.', 'info')
    return redirect('/')

if __name__ == "__main__":
    app.run(debug=True)