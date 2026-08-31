import json
import os
import uuid
import time
from datetime import datetime
from typing import Optional, Any

class SqlLogger:
    def __init__(self, log_path: str = "data/sql_queries.json"):
        self.log_path = log_path
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        """Ensures the JSON log file exists and is a valid JSON array."""
        directory = os.path.dirname(self.log_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", encoding="utf-8") as f:
                json.dump([], f)
        else:
            try:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if not isinstance(data, list):
                        raise ValueError("Root element is not a list")
            except (json.JSONDecodeError, ValueError) as e:
                # Do not silently erase! Backup and start fresh.
                backup_path = f"{self.log_path}.bak.{int(time.time())}"
                os.rename(self.log_path, backup_path)
                print(f"[SqlLogger] Malformed JSON detected. Backed up to {backup_path}. Starting fresh.")
                with open(self.log_path, "w", encoding="utf-8") as f:
                    json.dump([], f)

    def _get_query_type(self, sql: str) -> str:
        sql_upper = sql.strip().upper()
        for kw in ["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER"]:
            if sql_upper.startswith(kw):
                return kw
        return "OTHER"

    def log_query(
        self,
        request_id: str,
        question: str,
        sql: str,
        status: str,
        execution_time_ms: int,
        visualization_requested: bool,
        rows_returned: Optional[int] = None,
        error_message: Optional[str] = None
    ):
        """Appends a new SQL query log entry to the JSON file safely."""
        entry = {
            "id": str(uuid.uuid4()),
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "question": question,
            "sql": sql.strip(),
            "query_type": self._get_query_type(sql),
            "status": status,
            "execution_time_ms": execution_time_ms,
            "rows_returned": rows_returned,
            "visualization_requested": visualization_requested,
            "error": error_message
        }

        # Basic concurrency protection via retry backoff
        max_retries = 5
        for attempt in range(max_retries):
            try:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                data.append(entry)

                # Write to temp file then rename (atomic-ish on many OS)
                temp_path = self.log_path + ".tmp"
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4)
                os.replace(temp_path, self.log_path)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"[SqlLogger] Failed to write log after retries: {e}")
                else:
                    time.sleep(0.05 * (attempt + 1))
