from pathlib import Path
import os
from logger import get_logger

_LOGGER = get_logger(__file__)

PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
USER_DATA_DIRECTORY = os.environ.get("DATA_DIRECTORY")
DEFAULT_DATA_DIRECTORY = PROJECT_DIRECTORY / "data"
DATA_DIRECTORY = USER_DATA_DIRECTORY if USER_DATA_DIRECTORY else DEFAULT_DATA_DIRECTORY
WIKIPEDIA_DIRECTORY = DATA_DIRECTORY / "wikipedia"

_LOGGER.info(f"Using Data Directory: {DATA_DIRECTORY}")
