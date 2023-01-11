from src.paths import TEST_ARTICLES, WIKIPEDIA_PROCESSED_DIRECTORY
import json
from src.regex_collection import LINKS_REGEX, ARTICLE_TEMPLATE_REGEX
import re
from typing import List, Dict
from tqdm import tqdm


class SentencesExtractor:
    def __init__(self) -> None:
        self.redirects = self.get_redirects()

    @staticmethod
    def same_surface_entity(surface_entity_split):
        return len(surface_entity_split) == 1

    @staticmethod
    def different_surface_entity(surface_entity_split):
        return len(surface_entity_split) == 2

    @staticmethod
    def get_redirects() -> Dict[str, str]:
        redirects_collection = {}
        pbar = tqdm()
        pbar.set_description("Getting Redirects ...")
        for redirect_file in WIKIPEDIA_PROCESSED_DIRECTORY.glob("redirects*.json"):
            redirects = json.loads(redirect_file.read_text())
            redirects = {
                article["title"].lower(): article["text"].lower()
                for article in redirects
            }
            redirects_collection.update(redirects)
            pbar.update(1)
        pbar.close()
        return redirects_collection

    @staticmethod
    def link_to_string(surface: str, article: str) -> str:
        return f"({surface}) [{article}]"

    def format_link(self, link: str) -> str:
        link = link.replace("[", "").replace("]", "")
        surface_entity_split = link.split("|")
        if self.same_surface_entity(surface_entity_split):
            surface = surface_entity_split[0]
            return self.link_to_string(surface, surface)
        elif self.different_surface_entity(surface_entity_split):
            article, surface = surface_entity_split
            article = self.redirects.get(article.lower(), article)
            return self.link_to_string(surface, article)
        else:
            # not an article hyperlink e.g. File:1.jpg
            return ""

    def cleanup_article_text(self, text: str) -> str:
        text = re.sub(ARTICLE_TEMPLATE_REGEX, "", text)
        return re.sub(LINKS_REGEX, lambda link: self.format_link(link.group()), text)

    @staticmethod
    def cleanup_sentence(sentence: str) -> str:
        sentence = sentence.replace("'", "").replace('"', "")
        sentence = " ".join(sentence.split())
        return sentence

    def get_sentences(self, text: str) -> List[str]:
        """
        Intentionally using a simpler split approach,
        there will be some noise which could act as a regularizer
        """
        clean_article = self.cleanup_article_text(text)
        article_paragraphs = clean_article.split("\n")
        article_sentences = []
        for para in article_paragraphs:
            for sentence in para.split("."):
                sentence = self.cleanup_sentence(sentence)
                if not sentence:
                    continue
                article_sentences.append(sentence)
        return article_sentences


if __name__ == "__main__":
    # test code
    test_articles = json.loads(TEST_ARTICLES.read_text())
    sentences_extractor = SentencesExtractor()
    sentences = sentences_extractor.get_sentences(test_articles[1]["text"])
