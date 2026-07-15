# ruff: noqa: E501
import os
import sqlite3
import time
import uuid
from typing import Optional


class TelemetryTracer:
    """
    Tier-2 Logging: Writes 100% full granularity LLM prompts and responses to a SQLite database
    so we can forensically debug context poisoning and hallucination triggers.
    """
    def __init__(self, db_path: str = "logs/traces.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.run_id = str(uuid.uuid4())
        self._init_db()
        
    def _init_db(self) -> None:
        """Initializes the SQLite schema if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS traces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    step_number INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    tool_called TEXT,
                    input TEXT,
                    output TEXT,
                    is_error INTEGER DEFAULT 0,
                    timestamp REAL NOT NULL
                )
            """)
            try:
                cursor.execute("ALTER TABLE traces ADD COLUMN is_error INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass  # Column already exists
            # Create indices for faster lookups when querying a specific run
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_run_id ON traces (run_id)")
            conn.commit()

    def log_step(self, 
                 step_number: int, 
                 state: str, 
                 input_text: str, 
                 output_text: Optional[str], 
                 tool_called: Optional[str] = None,
                 is_error: bool = False) -> None:
        """Writes the raw text of a single step to the SQLite trace store."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO traces (
                    run_id, step_number, state, tool_called, input, output, is_error, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.run_id,
                step_number,
                state,
                tool_called,
                input_text,
                output_text,
                1 if is_error else 0,
                time.time()
            ))
            conn.commit()
            
    def get_run_id(self) -> str:
        return self.run_id
