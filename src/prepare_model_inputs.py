from typing import List
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer
from pathlib import Path
from config import config

from config import (
    MAX_SEQUENCE_LENGTH,
    RANDOM_SEED,
    TEST_SPLIT_RATIO,
    TRUNCATE_SENTENCES,
    VALIDATION_SPLIT_RATIO,
)
from paths import WIKIPEDIA_PROCESSED_DIRECTORY

WIKIPEDIA_TRAINING_FILES = [
    str(path) for path in list(WIKIPEDIA_PROCESSED_DIRECTORY.glob("train*.csv"))
]


class ModelInputs:
    def __init__(
        self,
        training_files: List[str],
        model_name: str,
        input_sentence_prefix: str,
    ) -> None:
        self.model_name = model_name
        self.input_sentence_prefix = input_sentence_prefix
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.training_files = training_files

    def get_train_dataset(self) -> DatasetDict:
        if not self.training_files:
            raise Exception(
                f"No Training Files Found at {WIKIPEDIA_PROCESSED_DIRECTORY}"
            )

        # load full dataset
        full_dataset = load_dataset("csv", data_files=self.training_files)

        # split it into train and test
        train_test_split = full_dataset["train"].train_test_split(
            test_size=TEST_SPLIT_RATIO, shuffle=True, seed=RANDOM_SEED
        )

        # split train further into train and validation
        train_validation_split = train_test_split["train"].train_test_split(
            test_size=VALIDATION_SPLIT_RATIO, shuffle=True, seed=RANDOM_SEED
        )

        # combine train, validation and test splits
        train_dataset = DatasetDict(
            {
                "train": train_validation_split["train"],
                "validation": train_validation_split["test"],
                "test": train_test_split["test"],
            }
        )
        return train_dataset

    def add_prefix_to_sentence(self, sentence: str) -> str:
        return f"{self.input_sentence_prefix}{sentence}"

    def add_prefix_and_tokenize(self, dataset: Dataset) -> Dataset:
        inputs = [self.add_prefix_to_sentence(sentence) for sentence in dataset["raw"]]
        targets = dataset["formatted"]
        input_target_tokenized = self.tokenizer(
            inputs,
            text_target=targets,
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=TRUNCATE_SENTENCES,
        )
        return input_target_tokenized

    def prepare(self, batched: bool = True) -> Dataset:
        train_dataset = self.get_train_dataset()
        model_inputs = train_dataset.map(self.add_prefix_and_tokenize, batched=batched)
        return model_inputs


def get_model_inputs(
    training_data_directory: str,
    model_name: str,
    input_sentence_prefix: str,
    training_files_prefix: str,
    training_files_extention: str,
) -> DatasetDict:
    training_files = [
        str(path)
        for path in list(
            training_data_directory.glob(
                f"{training_files_prefix}*.{training_files_extention}"
            )
        )
    ]
    dataset = ModelInputs(
        training_files=training_files,
        model_name=model_name,
        input_sentence_prefix=input_sentence_prefix,
    )

    return dataset
