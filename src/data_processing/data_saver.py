import os
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import re

logger = logging.getLogger(__name__)

class SaveData:
    def __init__(self, base_data_dir="data"):
        self.base_data_dir = Path(base_data_dir)
        self.raw_dir = self.base_data_dir / "raw"

    def ensure_directory_exists(self, dir_path):
        """Create directory if it doesn't exist"""
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {dir_path}")

