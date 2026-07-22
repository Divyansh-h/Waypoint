import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


class JSONFormatter(logging.Formatter):
    """
    Custom formatter to output logs in JSON format for structured logging.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_obj: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add exception info if present
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        # Include any extra kwargs passed to the logger
        # (e.g., logger.info("msg", extra={"file_count": 5}))
        for key, value in record.__dict__.items():
            if key not in [
                "args", "asctime", "created", "exc_info", "exc_text", "filename",
                "funcName", "id", "levelname", "levelno", "lineno", "module",
                "msecs", "message", "msg", "name", "pathname", "process",
                "processName", "relativeCreated", "stack_info", "thread", "threadName"
            ]:
                log_obj[key] = value

        return json.dumps(log_obj)


def setup_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """
    Sets up a structured JSON logger that outputs to both console and a log file.
    
    Args:
        name: Name of the logger.
        log_dir: Directory to save the log files.
        
    Returns:
        logging.Logger: The configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Prevent adding handlers multiple times if instantiated repeatedly
    if logger.handlers:
        return logger

    # Ensure log directory exists
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Create handlers
    console_handler = logging.StreamHandler(sys.stdout)
    
    # Create a unique log file for this run based on timestamp
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(log_path / f"{name}_run_{run_timestamp}.log")
    
    # Set the JSON formatter for both
    json_formatter = JSONFormatter()
    console_handler.setFormatter(json_formatter)
    file_handler.setFormatter(json_formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger
