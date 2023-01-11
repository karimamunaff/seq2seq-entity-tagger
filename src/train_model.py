from transformers import AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from src.paths import WIKIPEDIA_PROCESSED_DIRECTORY, MODELS_DIRECTORY
from src.prepare_model_inputs import get_model_inputs
from transformers import DataCollatorForSeq2Seq
from src.config import config


def train() -> None:
    model_inputs = get_model_inputs(
        training_data_directory=WIKIPEDIA_PROCESSED_DIRECTORY,
        model_name=config["train"]["model_name"],
        input_sentence_prefix=config["data"]["input_sentence_prefix"],
        training_files_prefix=config["data"]["training_filenames_prefix"],
        training_files_extention=config["data"]["training_filenames_extention"],
        training_num_articles=config["data"]["wikipedia_articles_limit"],
    )
    training_dataset = model_inputs.prepare()

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=model_inputs.tokenizer,
        model=config["train"]["model_name"],
        padding=config["train"]["pad_batch"],
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=MODELS_DIRECTORY,
        evaluation_strategy=config["train"]["evaluation_strategy"],
        learning_rate=float(config["train"]["learning_rate"]),
        per_device_train_batch_size=int(config["train"]["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(config["train"]["per_device_eval_batch_size"]),
        weight_decay=float(config["train"]["weight_decay"]),
        save_total_limit=int(config["train"]["save_total_limit"]),
        num_train_epochs=int(config["train"]["num_train_epochs"]),
        use_mps_device=bool(int(config["train"]["use_mps_device"])),
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(model_inputs.model_name)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=training_dataset["train"],
        eval_dataset=training_dataset["test"],
        tokenizer=model_inputs.tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    train()
