
USER_DATA_DIRECTORY ?= data/
MAX_WIKIPEDIA_ARTICLES ?= 10000
SAVE_JSON_EVERY ?= 1000

.PHONY: setup_project
setup_project:
	poetry install

.PHONY: extract_wikipedia
extract_wikipedia:
	poetry run python src/extract_wikipedia.py --save-every=$(SAVE_JSON_EVERY) --max-articles=$(MAX_WIKIPEDIA_ARTICLES)



