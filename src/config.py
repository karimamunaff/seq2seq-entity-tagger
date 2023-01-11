from src.paths import CONFIG_PATH
import json
import os
from typing import Dict


def _get_current_wikipedia_date() -> int:
    wikipedia_date = os.environ.get("WIKIPEDIA_DUMP_DATE")
    if wikipedia_date is None:
        raise Exception(
            "Wikipedia data not found."
            "Dump Date needs to be specified via env variable WIKIPEDIA_DUMP_DATE"
            "Or Set it in Makefile at project root folder"
        )
    return int(wikipedia_date)


def _update_wikipedia_dumpdate() -> Dict:
    wikipedia_date = _get_current_wikipedia_date()
    config = json.loads(CONFIG_PATH.read_text())
    if int(config["data"]["wikipedia_dump_date"]) != wikipedia_date:
        config["data"]["wikipedia_dump_date"] = wikipedia_date
        with open(CONFIG_PATH, "w+") as f:
            json.dump(config, f)
    return config


config = _update_wikipedia_dumpdate()
