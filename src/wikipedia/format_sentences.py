import json
import re
from typing import List, Union

import pandas
import typer

from paths import WIKIPEDIA_PROCESSED_DIRECTORY
from extract_sentences import SentencesExtractor
from regex_collection import FORMATTED_ENTITY_REGEX
from logger import get_logger

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


def save_as_csv(
    max_articles: int = 100,
    training_file_prefix: str = "train",
    training_file_extension: str = "csv",
) -> None:
    save_filename = (
        WIKIPEDIA_PROCESSED_DIRECTORY
        / f"{training_file_prefix}_{max_articles}.{training_file_extension}"
    )
    articles_sentences_collection = extract_article_sentences(max_articles)
    training_data = pandas.DataFrame()
    training_data["formatted"] = articles_sentences_collection
    training_data["raw"] = training_data["formatted"].apply(convert_sentence_toraw)
    if training_file_extension == "csv":
        training_data.to_csv(save_filename, index=False)
    elif training_file_extension == "parquet":
        training_data.to_parquet(save_filename, index=False)
    else:
        raise Exception("Files should have .csv or .parquet extention")
    _LOGGER.info(f"Done. Training data saved at {save_filename}")


if __name__ == "__main__":
    typer.run(save_as_csv)
