from pathlib import Path
import os
from logger import get_logger

_LOGGER = get_logger(__file__)

PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
USER_DATA_DIRECTORY = os.environ.get("USER_DATA_DIRECTORY")
DEFAULT_DATA_DIRECTORY = PROJECT_DIRECTORY / "data"
DATA_DIRECTORY = (
    Path(USER_DATA_DIRECTORY) if USER_DATA_DIRECTORY else DEFAULT_DATA_DIRECTORY
)

WIKIPEDIA_DIRECTORY = DATA_DIRECTORY / "wikipedia"
WIKIPEDIA_BZ2_XML_FILE = WIKIPEDIA_DIRECTORY / "enwiki-20220701-pages-articles.xml.bz2"
WIKIPEDIA_PROCESSED_DIRECTORY = WIKIPEDIA_DIRECTORY / "processed"
WIKIPEDIA_PROCESSED_DIRECTORY.mkdir(exist_ok=True)

TEST_DATA_DIRECTORY = PROJECT_DIRECTORY / "tests" / "data"
TEST_ARTICLES = TEST_DATA_DIRECTORY / "test_articles.json"

MODELS_DIRECTORY = DATA_DIRECTORY / "models"
MODELS_DIRECTORY.mkdir(exist_ok=True)

_LOGGER.info(f"Using Data Directory: {DATA_DIRECTORY}")
