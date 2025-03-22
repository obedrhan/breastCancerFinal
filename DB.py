import sqlite3

def create_database():
    # Connect to SQLite database (creates file if it doesn't exist)
    conn = sqlite3.connect('radiologist_system.db')
    cursor = conn.cursor()

    # Create Radiologists table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Radiologists (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fullname TEXT NOT NULL,
            diploma_number TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')

    # Create Patients table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            radiologist_id INTEGER NOT NULL,
            fullname TEXT NOT NULL,
            patient_id TEXT UNIQUE NOT NULL,
            dob DATE NOT NULL,
            FOREIGN KEY (radiologist_id) REFERENCES Radiologists (id)
        )
    ''')

    # Create ClassificationResults table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ClassificationResults (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            radiologist_id INTEGER NOT NULL,
            classification_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            FOREIGN KEY (patient_id) REFERENCES Patients (id),
            FOREIGN KEY (radiologist_id) REFERENCES Radiologists (id)
        )
    ''')

    # Commit changes and close the connection
    conn.commit()
    conn.close()
    print("Database and tables created successfully.")

if __name__ == "__main__":
    create_database()