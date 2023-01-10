
USER_DATA_DIRECTORY ?= data/
MAX_WIKIPEDIA_ARTICLES ?= 10000
SAVE_JSON_EVERY ?= 1000
TRAINING_FILE_PREFIX ?= train
TRAINING_FILE_EXTENTION ?= csv

MODEL_NAME ?= t5-small
INPUT_SENTENCE_PREFIX ?= "Tag Entities: "
TRAINING_DATA_DIRECTORY ?= $(USER_DATA_DIRECTORY)

.PHONY: setup_project
setup_project:
	poetry install

.PHONY: wikipedia/extract_articles
wikipedia/extract_articles:
	poetry run python src/wikipedia/extract_articles.py --save-every=$(SAVE_JSON_EVERY) --max-articles=$(MAX_WIKIPEDIA_ARTICLES)

.PHONY: wikipedia/format_sentences
wikipedia/format_sentences:
	poetry run python src/wikipedia/format_sentences.py --max-articles=$(MAX_WIKIPEDIA_ARTICLES) --training-file-prefix=$(TRAINING_FILE_PREFIX) --training-file-extention=$(TRAINING_FILE_EXTENTION)

.PHONY: prepare_model_inputs
prepare_model_inputs:
	poetry run python src/prepare_model_inputs.py --training-file-prefix=$(TRAINING_FILE_PREFIX) --training-file-extention=$(TRAINING_FILE_EXTENTION) --model-name=$(MODEL_NAME) --input-sentence-prefix=$(INPUT_SENTENCE_PREFIX) --training_data_directory=$(TRAINING_DATA_DIRECTORY)
