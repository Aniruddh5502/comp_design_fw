import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logger(name="comp_design_fw", log_dir="logs"):
    """Configures a logger that outputs to both console and file."""
    
    # Ensure log directory exists
    root_dir = Path(__file__).parent
    log_path = root_dir / log_dir
    log_path.mkdir(exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers if called multiple times
    if not logger.handlers:
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(module)s:%(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File Handler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"framework_{timestamp}.log"
        file_handler = logging.FileHandler(log_path / file_name)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# Global logger instance
logger = setup_logger()
