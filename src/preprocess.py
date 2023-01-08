from paths import TEST_ARTICLES
import json
from regex_collection import LINKS_REGEX
import re
from typing import List


def same_surface_entity(surface_entity_split):
    return len(surface_entity_split) == 1


def different_surface_entity(surface_entity_split):
    return len(surface_entity_split) == 2


def get_links(text: str) -> List[str]:
    links_raw = re.findall(LINKS_REGEX, text)
    surface_to_entity = {}
    hard2split_links = []
    for link in links_raw:
        surface_entity_split = link.split("|")
        if same_surface_entity(surface_entity_split):
            surface = surface_entity_split[0]
            surface_to_entity[surface] = surface
        elif different_surface_entity(surface_entity_split):
            article, surface = surface_entity_split
            # todo: need to map article using redirects
            surface_to_entity[surface] = article
        else:
            hard2split_links.append(link)
    return surface_to_entity, hard2split_links


if __name__ == "__main__":
    # test code
    test_articles = json.loads(TEST_ARTICLES.read_text())
    surface_to_entity, hard2split_links = get_links(test_articles[1]["text"])
