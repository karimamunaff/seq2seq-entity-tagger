from datasets import load_dataset, DatasetDict, Dataset
from paths import WIKIPEDIA_PROCESSED_DIRECTORY
from transformers import AutoTokenizer
from config import (
    RANDOM_SEED,
    TEST_SPLIT_RATIO,
    VALIDATION_SPLIT_RATIO,
    MAX_SEQUENCE_LENGTH,
    TRUNCATE_SENTENCES,
)

T5_ENTITY_PREFIX = "Tag Entities: "
T5_MODEL = "t5-small"


class Tokenizer:
    def __init__(
        self, t5_model: str = "t5-small", t5_prefix: str = "Tag Entities: "
    ) -> None:
        self.t5_model = t5_model
        self.t5_prefix = t5_prefix
        self.tokenizer = AutoTokenizer.from_pretrained(self.t5_model)
        self.training_files = [
            str(path) for path in list(WIKIPEDIA_PROCESSED_DIRECTORY.glob("train*.csv"))
        ]

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
        return f"{self.t5_prefix}{sentence}"

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

    def get_model_inputs(self, batched: bool = True) -> Dataset:
        train_dataset = self.get_train_dataset()
        model_inputs = train_dataset.map(self.add_prefix_and_tokenize, batched=batched)
        return model_inputs


if __name__ == "__main__":
    # example code
    dataset = Tokenizer().get_model_inputs()
