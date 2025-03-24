import sqlite3

def create_database():
    conn = sqlite3.connect('radiologist_system.db')
    cursor = conn.cursor()

    # Radiologists table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Radiologists (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fullname TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Patients table (national_id as PRIMARY KEY)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Patients (
            national_id TEXT PRIMARY KEY,
            radiologist_id INTEGER NOT NULL,
            fullname TEXT NOT NULL,
            dob TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (radiologist_id) REFERENCES Radiologists (id) ON DELETE CASCADE
        )
    ''')

    # Classification Results table (linked to national_id now)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ClassificationResults (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            national_id TEXT NOT NULL,
            radiologist_id INTEGER NOT NULL,
            classification_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            prediction TEXT NOT NULL CHECK (prediction IN ('BENIGN', 'MALIGNANT')),
            confidence REAL NOT NULL,
            FOREIGN KEY (national_id) REFERENCES Patients (national_id) ON DELETE CASCADE,
            FOREIGN KEY (radiologist_id) REFERENCES Radiologists (id) ON DELETE CASCADE
        )
    ''')

    conn.commit()
    conn.close()
    print("✅ Database updated successfully.")

if __name__ == "__main__":
    create_database()