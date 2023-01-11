
WIKIPEDIA_DUMP_DATE ?= 20220701
USER_DATA_DIRECTORY ?= data/
USER_MODELS_DIRECTORY ?= models/

MODEL_NAME ?= t5-small
INPUT_SENTENCE_PREFIX ?= "Tag Entities: "
TRAINING_DATA_DIRECTORY ?= $(USER_DATA_DIRECTORY)

.PHONY: setup_project
setup_project:
	poetry install

.PHONY wikipedia/download
wikipedia/download:
	wget https://dumps.wikimedia.org/enwiki/${WIKIPEDIA_DUMP_DATE}/enwiki-$(WIKIPEDIA_DUMP_DATE)-pages-articles.xml.bz2 -P $(USER_DATA_DIRECTORY)

.PHONY: wikipedia/extract_articles
wikipedia/extract_articles:
	poetry run python src/wikipedia/extract_articles.py

.PHONY: wikipedia/format_sentences
wikipedia/format_sentences:
	poetry run python src/wikipedia/format_sentences.py

.PHONY: train
train:
	poetry run python src/train_model.py