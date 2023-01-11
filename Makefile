
WIKIPEDIA_DUMP_DATE ?= 20220701
USER_DATA_DIRECTORY ?= data/
USER_MODELS_DIRECTORY ?= models/

MODEL_NAME ?= t5-small
INPUT_SENTENCE_PREFIX ?= "Tag Entities: "
TRAINING_DATA_DIRECTORY ?= $(USER_DATA_DIRECTORY)

.PHONY: .env
.env:
	@bash -c "echo 'PYTHONPATH=${PWD}:${PYTHONPATH}/src' > .env"

.PHONY: setup_project
setup_project: .env
	#poetry install

.PHONY: wikipedia/download
wikipedia/download:
	wget https://dumps.wikimedia.org/enwiki/${WIKIPEDIA_DUMP_DATE}/enwiki-$(WIKIPEDIA_DUMP_DATE)-pages-articles.xml.bz2 -P $(USER_DATA_DIRECTORY)

.PHONY: wikipedia/extract_articles
wikipedia/extract_articles:
	WIKIPEDIA_DUMP_DATE=$(WIKIPEDIA_DUMP_DATE) poetry run python src/wikipedia/extract_articles.py

.PHONY: wikipedia/format_sentences
wikipedia/format_sentences:
	WIKIPEDIA_DUMP_DATE=$(WIKIPEDIA_DUMP_DATE) poetry run python3 -m src.wikipedia.format_sentences

.PHONY: train
train:
	WIKIPEDIA_DUMP_DATE=$(WIKIPEDIA_DUMP_DATE) USER_MODELS_DIRECTORY=$(USER_MODELS_DIRECTORY) poetry run python3 -m src.train_model