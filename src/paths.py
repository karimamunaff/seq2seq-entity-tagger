from pathlib import Path
import os
from logger import get_logger

_LOGGER = get_logger(__file__)

PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
USER_DATA_DIRECTORY = os.environ.get("DATA_DIRECTORY")
DEFAULT_DATA_DIRECTORY = PROJECT_DIRECTORY / "data"
DATA_DIRECTORY = USER_DATA_DIRECTORY if USER_DATA_DIRECTORY else DEFAULT_DATA_DIRECTORY

WIKIPEDIA_DIRECTORY = DATA_DIRECTORY / "wikipedia"
WIKIPEDIA_XML_FILE = WIKIPEDIA_DIRECTORY / "enwiki-20220701-pages-articles.xml.bz2"
WIKIPEDIA_PROCESSED_DIRECTORY = WIKIPEDIA_DIRECTORY / "processed"
WIKIPEDIA_PROCESSED_DIRECTORY.mkdir(exist_ok=True)

_LOGGER.info(f"Using Data Directory: {DATA_DIRECTORY}")
