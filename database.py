import sqlite3
import hashlib

DB_NAME = "autoinsight.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # User Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    # Analysis History Table linked strictly to user_email
    c.execute('''
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT,
            dataset_name TEXT,
            profiling_result TEXT,
            forecast_result TEXT,
            report_result TEXT,
            keyword_result TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_email) REFERENCES users (email)
        )
    ''')
    conn.commit()
    conn.close()

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(email: str, password: str) -> bool:
    init_db()
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (email, password) VALUES (?, ?)", (email.lower(), hash_password(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def authenticate_user(email: str, password: str) -> bool:
    init_db()
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE email = ?", (email.lower(),))
    record = c.fetchone()
    conn.close()
    if record and record[0] == hash_password(password):
        return True
    return False

def save_analysis(user_email: str, dataset_name: str, prof_res: str, fore_res: str, rep_res: str, kw_res: str):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        INSERT INTO analysis_history (user_email, dataset_name, profiling_result, forecast_result, report_result, keyword_result)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (user_email.lower(), dataset_name, prof_res, fore_res, rep_res, kw_res))
    conn.commit()
    conn.close()

def get_user_history(user_email: str):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        SELECT dataset_name, created_at, profiling_result, forecast_result, report_result, keyword_result
        FROM analysis_history 
        WHERE user_email = ? 
        ORDER BY created_at DESC
    ''', (user_email.lower(),))
    rows = c.fetchall()
    conn.close()
    return rows
