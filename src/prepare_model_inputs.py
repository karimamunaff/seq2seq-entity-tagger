from typing import List
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer
from src.paths import WIKIPEDIA_PROCESSED_DIRECTORY
from src.config import config
from src.logger import get_logger

_LOGGER = get_logger(__file__)


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

    def get_train_dataset(
        self,
        test_split_ratio: float = config["train"]["test_split_ratio"],
        validation_split_ratio: float = config["train"]["validation_split_ratio"],
        dataset_file_extention: str = config["data"]["training_filenames_extention"],
        random_seed: int = config["general"]["random_seed"],
    ) -> DatasetDict:
        if not self.training_files:
            raise Exception(
                f"No Training Files Found at {WIKIPEDIA_PROCESSED_DIRECTORY}"
            )

        # load full dataset
        full_dataset = load_dataset(
            dataset_file_extention, data_files=self.training_files
        )

        # split it into train and test
        train_test_split = full_dataset["train"].train_test_split(
            test_size=test_split_ratio,
            shuffle=True,
            seed=random_seed,
        )

        # split train further into train and validation
        train_validation_split = train_test_split["train"].train_test_split(
            test_size=validation_split_ratio,
            shuffle=True,
            seed=random_seed,
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
            max_length=config["train"]["max_sequence_length"],
            truncation=config["train"]["truncate_sentences"],
        )
        return input_target_tokenized

    def prepare(self, batched: bool = True) -> Dataset:
        train_dataset = self.get_train_dataset()
        model_inputs = train_dataset.map(self.add_prefix_and_tokenize, batched=batched)
        return model_inputs


def get_model_inputs(
    model_name: str,
    input_sentence_prefix: str,
    training_data_directory: str,
    training_num_articles: int,
    training_files_prefix: str,
    training_files_extention: str,
) -> DatasetDict:
    training_files = [
        str(path)
        for path in list(
            training_data_directory.glob(
                f"{training_files_prefix}_{training_num_articles}*.{training_files_extention}"
            )
        )
    ]
    _LOGGER.info(f"Using Training Files: {' '.join(training_files)}")
    dataset = ModelInputs(
        training_files=training_files,
        model_name=model_name,
        input_sentence_prefix=input_sentence_prefix,
    )

    return dataset


if __name__ == "__main__":
    from src.config import config

    model_inputs = get_model_inputs(
        training_data_directory=WIKIPEDIA_PROCESSED_DIRECTORY,
        model_name=config["train"]["model_name"],
        input_sentence_prefix=config["data"]["input_sentence_prefix"],
        training_files_prefix=config["data"]["training_filenames_prefix"],
        training_files_extention=config["data"]["training_filenames_extention"],
        training_num_articles=config["data"]["wikipedia_articles_limit"],
    )
