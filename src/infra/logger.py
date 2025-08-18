import logging
import os
from datetime import datetime
from typing import Optional


class LoggerManager:
    """Singleton class to manage the logger configuration."""

    _configured = False
    _log_file = ""

    @classmethod
    def configure(cls, level: int = logging.INFO, log_dir: str = "logs") -> None:
        # Configure the logger with the specified level and directory.
        if cls._configured:
            return

        os.makedirs("logs", exist_ok=True)
        cls._log_file = os.path.join(
            log_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(cls._log_file, encoding="utf-8"),
                logging.StreamHandler(),
            ],
        )

        cls._configured = True
        logging.info("Logger configured successfully.")

    @classmethod
    def get_logger(cls, name: Optional[str] = None) -> logging.Logger:
        # Return a logger instance with the specified name.
        if not cls._configured:
            cls.configure()

        return logging.getLogger(name or __name__)

    @classmethod
    def get_current_log_file(cls) -> str:
        # Return the path to the current log file.
        if not cls._configured:
            raise RuntimeError("Logger is not configured. Call configure() first.")
        return cls._log_file
