
USER_DATA_DIRECTORY ?= data/
MAX_WIKIPEDIA_ARTICLES ?= 10000

.PHONY: setup_project
setup_project:
	poetry install


