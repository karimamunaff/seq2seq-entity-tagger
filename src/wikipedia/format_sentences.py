import json
import re
from typing import List, Union

import pandas

from src.paths import WIKIPEDIA_PROCESSED_DIRECTORY
from src.wikipedia.extract_sentences import SentencesExtractor
from src.regex_collection import FORMATTED_ENTITY_REGEX
from logger import get_logger
from config import config

_LOGGER = get_logger(__file__)


def iterate_article_texts(max_articles: int = 100) -> Union[None, str]:
    total_articles_read = 0
    for articles_json in WIKIPEDIA_PROCESSED_DIRECTORY.glob("articles*.json"):
        articles_collection = json.loads(articles_json.read_text())
        for article in articles_collection:
            if total_articles_read > max_articles:
                return
            total_articles_read += 1
            yield article["text"]


def extract_article_sentences(max_articles: int = 100) -> List[str]:
    sentences_extractor = SentencesExtractor()
    article_sentences_collection = []
    for article_text in iterate_article_texts(max_articles):
        if article_text is None:
            break
        article_sentences = sentences_extractor.get_sentences(article_text)
        article_sentences_collection.extend(article_sentences)
    return article_sentences_collection


def convert_sentence_toraw(sentence: str) -> str:
    sentence_entities_removed = re.sub(FORMATTED_ENTITY_REGEX, "", sentence)
    sentence_nocircular_brackets = (
        sentence_entities_removed.replace("(", "").replace(")", "").strip()
    )
    sentence_noextra_whitespaces = " ".join(sentence_nocircular_brackets.split())
    return sentence_noextra_whitespaces


def is_bad_sentence(
    sentence: str,
    bad_characters: List[str] = "{|[]",
    min_sentence_length: int = 5,
    max_sentence_length: int = 128,
) -> str:
    num_words = len(sentence.split())
    return (
        any(character in bad_characters for character in sentence)
        or num_words <= min_sentence_length
        or num_words >= max_sentence_length
    )


def filter_bad_sentences(training_data: pandas.DataFrame) -> pandas.DataFrame:
    return training_data[~training_data["raw"].apply(is_bad_sentence)]


def save_as_csv(
    training_filename_prefix: str = "train",
    training_filenames_extention: str = "csv",
    max_articles: int = 100,
) -> None:
    save_filename = (
        WIKIPEDIA_PROCESSED_DIRECTORY
        / f"{training_filename_prefix}_{max_articles}.{training_filenames_extention}"
    )
    articles_sentences_collection = extract_article_sentences(max_articles)
    training_data = pandas.DataFrame()
    training_data["formatted"] = articles_sentences_collection
    _LOGGER.info(f"Filtering Training data file {save_filename}")
    training_data["raw"] = training_data["formatted"].apply(convert_sentence_toraw)
    _LOGGER.info(f"Rows after filtering {len(training_data)}")
    training_data = filter_bad_sentences(training_data)
    if training_filenames_extention == "csv":
        training_data.to_csv(save_filename, index=False)
    elif training_filenames_extention == "parquet":
        training_data.to_parquet(save_filename, index=False)
    else:
        raise Exception("Files should have .csv or .parquet extention")
    _LOGGER.info(f"Done. Training data saved at {save_filename}")


if __name__ == "__main__":
    save_as_csv(
        training_filename_prefix=config["data"]["training_filenames_prefix"],
        training_filenames_extention=config["data"]["training_filenames_extention"],
        max_articles=config["data"]["wikipedia_articles_limit"],
    )
