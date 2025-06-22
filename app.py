import os
import sqlite3
import re
import uuid
from datetime import datetime, timedelta
from flask import (
    Flask, request, session, jsonify,
    send_from_directory, url_for
)
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from skimage.io import imsave

from segmentation_feature import (
    load_dicom_as_image, contrast_enhancement,
    region_growing, morphological_operations,
    contour_extraction, crop_image_with_contours
)
from new_predict import predict_mammogram

# React build klasörünün yolu
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
REACT_BUILD = os.path.join(BASE_DIR, 'frontend', 'dist')

# Flask uygulaması: static_folder ve static_url_path ayarlandı
app = Flask(__name__, static_folder=REACT_BUILD, static_url_path='')
app.secret_key = 'your_secret_key'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=1)
# React ile aynı site içinde çerez paylaşımı için
app.config['SESSION_COOKIE_SAMESITE'] = 'None'
app.config['SESSION_COOKIE_SECURE'] = True

CORS(app, supports_credentials=True)

# .dcm dosyalarının yükleneceği klasör
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'dcm'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# === Auth Routes ===
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    fullname = data.get('fullname', '').strip()
    email = data.get('email', '').strip()
    password = data.get('password', '').strip()

    if not fullname or not email or not password:
        return jsonify(error="Missing fields"), 400

    pw_regex = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\\d).{8,}$'
    if not re.match(pw_regex, password):
        return jsonify(error="Password must be 8+ characters with upper, lower, number"), 400

    conn = sqlite3.connect('radiologist_system.db')
    c = conn.cursor()
    c.execute('SELECT 1 FROM Radiologists WHERE email=?', (email,))
    if c.fetchone():
        conn.close()
        return jsonify(error="Email already registered"), 400

    hashed = generate_password_hash(password, method='pbkdf2:sha256')
    c.execute('INSERT INTO Radiologists (fullname, email, password) VALUES (?, ?, ?)',
              (fullname, email, hashed))
    conn.commit()
    conn.close()
    return jsonify(message="Registered"), 200

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email', '').strip()
    password = data.get('password', '').strip()

    if not email or not password:
        return jsonify(error="Missing credentials"), 400

    conn = sqlite3.connect('radiologist_system.db')
    c = conn.cursor()
    c.execute('SELECT id, password FROM Radiologists WHERE email=?', (email,))
    row = c.fetchone()
    conn.close()

    if not row or not check_password_hash(row[1], password):
        return jsonify(error="Invalid email or password"), 401

    session.permanent = True
    session['user_id'] = row[0]
    return jsonify(message="Logged in"), 200

@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify(message="Logged out successfully"), 200

# === Mammogram Upload & Prediction ===
@app.route('/process', methods=['POST'])
def process():
    if 'user_id' not in session:
        return jsonify(error="Unauthorized"), 403

    try:
        patient_name = request.form.get('patient_name', '').strip()
        national_id = request.form.get('national_id', '').strip()
        file = request.files.get('file')

        if not patient_name or not national_id or not file or file.filename == '':
            return jsonify(error="Missing required fields"), 400

        if not allowed_file(file.filename):
            return jsonify(error="Invalid file type. Only .dcm allowed."), 400

        filename = secure_filename(file.filename)
        full_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(full_path)

        # Segmentation
        orig = load_dicom_as_image(full_path)
        enh = contrast_enhancement(orig)
        seed = (enh.shape[1] // 2, enh.shape[0] // 2)
        grown = region_growing(enh, seed, threshold=15)
        refined = morphological_operations(grown)
        _, ctrs = contour_extraction(refined)
        cropped = crop_image_with_contours(orig, ctrs)

        # Save segmented image
        seg_fn = f"segmented_{uuid.uuid4().hex}.png"
        seg_path = os.path.join(app.config['UPLOAD_FOLDER'], seg_fn)
        imsave(seg_path, cropped)

        # Prediction
        result = predict_mammogram(full_path, cropped)

        # Veritabanı işlemleri
        conn = sqlite3.connect('radiologist_system.db')
        c = conn.cursor()
        c.execute('SELECT 1 FROM Patients WHERE national_id=?', (national_id,))
        if not c.fetchone():
            c.execute('''
                INSERT INTO Patients (national_id, fullname, radiologist_id, dob)
                VALUES (?, ?, ?, ?)
            ''', (national_id, patient_name, session['user_id'], datetime.now().strftime('%Y-%m-%d')))
            conn.commit()

        c.execute('''
            INSERT INTO ClassificationResults
            (national_id, radiologist_id, prediction, confidence)
            VALUES (?, ?, ?, ?)
        ''', (national_id, session['user_id'], result["prediction"].upper(), result["confidence"] * 100))
        conn.commit()
        conn.close()

        return jsonify({
            "prediction": result["prediction"],
            "confidence": round(result["confidence"] * 100, 2),
            "image_url": url_for('uploaded_file', filename=seg_fn, _external=True)
        }), 200

    except Exception as e:
        print("🔥 Error in /process:", e)
        return jsonify(error=str(e)), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/view_previous', methods=['POST'])
def view_previous():
    if 'user_id' not in session:
        return jsonify(error="Unauthorized"), 403

    data = request.get_json()
    patient_name = data.get('patient_name', '').strip()

    conn = sqlite3.connect('radiologist_system.db')
    c = conn.cursor()
    if patient_name:
        c.execute('''
            SELECT CR.classification_date, CR.prediction, CR.confidence, P.fullname
            FROM ClassificationResults CR
            JOIN Patients P ON CR.national_id = P.national_id
            WHERE CR.radiologist_id = ? AND P.fullname LIKE ?
            ORDER BY CR.classification_date DESC
        ''', (session['user_id'], f"%{patient_name}%"))
    else:
        c.execute('''
            SELECT CR.classification_date, CR.prediction, CR.confidence, P.fullname
            FROM ClassificationResults CR
            JOIN Patients P ON CR.national_id = P.national_id
            WHERE CR.radiologist_id = ?
            ORDER BY CR.classification_date DESC
        ''', (session['user_id'],))
    rows = c.fetchall()
    conn.close()

    return jsonify([
        {
            "date": row[0],
            "prediction": row[1],
            "confidence": round(row[2], 2),
            "fullname": row[3]
        }
        for row in rows
    ])

# React SPA catch-all
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react(path):
    full_path = os.path.join(REACT_BUILD, path)
    if path and os.path.exists(full_path):
        return send_from_directory(REACT_BUILD, path)
    return send_from_directory(REACT_BUILD, 'index.html')

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
