import json
import xml.etree.ElementTree as etree
from bz2 import BZ2File
from dataclasses import dataclass
from typing import Dict, List

from tqdm import tqdm

from logger import get_logger
from src.config import config
from src.paths import WIKIPEDIA_BZ2_XML_FILE, WIKIPEDIA_PROCESSED_DIRECTORY

_LOGGER = get_logger(__file__)


@dataclass
class WikipediaArticle:
    title: str
    text: str
    is_redirect: bool


def cleanup_tag(tag: str) -> str:
    cleanup_till_index = tag.rfind("}")
    return tag[cleanup_till_index + 1 :] if cleanup_till_index != -1 else tag


def iterate_articles(xmlfile: BZ2File) -> WikipediaArticle:
    current_title = ""
    for event, content in etree.iterparse(xmlfile, events=("start", "end")):
        if event == "start":
            continue
        tag = cleanup_tag(content.tag)

        # title
        if tag == "title":
            current_title = content.text

        # article text
        elif (
            tag == "text"
            and (content.text is not None)
            and (not content.text.startswith("#REDIRECT"))
        ):
            yield WikipediaArticle(current_title, content.text, False)

        # redirect
        elif tag == "redirect":
            yield WikipediaArticle(current_title, content.attrib["title"], True)


def save_articles(
    articles_collection: List[Dict], save_index: int, save_prefix: str = "articles"
) -> None:
    save_filename = WIKIPEDIA_PROCESSED_DIRECTORY / f"{save_prefix}_{save_index}.json"
    _LOGGER.info(f"saving to file {save_filename}")
    with open(save_filename, "w+") as f:
        json.dump(articles_collection, f)


def save_intermediate(
    articles_collection: List[Dict], redirects_collection: List[Dict], save_index: int
) -> None:
    save_articles(articles_collection, save_index)
    save_articles(
        redirects_collection,
        save_index,
        save_prefix="redirects",
    )


def extract_wikipedia_dump(save_every: int = 1000) -> None:
    progress_bar = tqdm()
    progress_bar.set_description("Processing Wikipedia Articles ...")
    articles_collection = []
    redirects_collection = []
    with BZ2File(WIKIPEDIA_BZ2_XML_FILE, "r") as xmlfile:
        for index, article in tqdm(enumerate(iterate_articles(xmlfile))):
            if article.is_redirect:
                redirects_collection.append(article.__dict__)
            else:
                articles_collection.append(article.__dict__)
            if index % save_every == 0 and index > 0:
                _LOGGER.info(f"Processed {index} wikipedia articles")
                save_intermediate(
                    articles_collection,
                    redirects_collection,
                    save_index=int(index / save_every),
                )
                articles_collection = []
                redirects_collection = []
            progress_bar.update(1)
    progress_bar.close()


def extract():
    extract_wikipedia_dump(
        save_every=config["data"]["articles_per_shard"],
    )


if __name__ == "__main__":
    extract()
