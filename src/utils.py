"""
Utility Functions
=================
Logging, CSV handling, and SQLite database operations.
"""

import pandas as pd
import os
import sqlite3
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('LLM-Prompt-Evaluator')


def get_logger(name):
    return logging.getLogger(name)


def read_prompts_csv(filepath):
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    return pd.DataFrame()


def save_results_csv(results, filepath):
    df = pd.DataFrame(results)
    if os.path.exists(filepath):
        existing_df = pd.read_csv(filepath)
        df = pd.concat([existing_df, df], ignore_index=True)
    df.to_csv(filepath, index=False)


# =====================
# Database Functions
# =====================

def init_db(db_path):
    """Create the database and tables if they don't exist."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS evaluation_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt TEXT,
            expected_output TEXT,
            llm_output TEXT,
            model_name TEXT,
            score REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()

    # Upgrade schema: add new columns if they don't exist
    # This keeps backward compatibility with older databases
    new_columns = [
        ("judge_score", "REAL"),
        ("feedback", "TEXT"),
        ("semantic_similarity", "REAL"),
        ("comparison_run_id", "TEXT"),
    ]
    for col_name, col_type in new_columns:
        try:
            cursor.execute(f"ALTER TABLE evaluation_results ADD COLUMN {col_name} {col_type}")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists, skip

    conn.close()


def save_to_db(db_path, prompt, expected_output, llm_output, model_name, score,
               judge_score=None, feedback=None, semantic_similarity=None, comparison_run_id=None):
    """Save a single evaluation result to the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO evaluation_results 
        (prompt, expected_output, llm_output, model_name, score, judge_score, feedback, semantic_similarity, comparison_run_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (prompt, expected_output, llm_output, model_name, score, judge_score, feedback, semantic_similarity, comparison_run_id))
    conn.commit()
    conn.close()


def get_history(db_path, limit=100):
    """
    Fetch past evaluation results from the database.
    Returns a list of dictionaries, newest first.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, prompt, expected_output, llm_output, model_name, score, 
               judge_score, feedback, semantic_similarity, comparison_run_id, timestamp
        FROM evaluation_results 
        ORDER BY timestamp DESC 
        LIMIT ?
    ''', (limit,))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def clear_history(db_path):
    """Delete all evaluation results from the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM evaluation_results")
    conn.commit()
    conn.close()
