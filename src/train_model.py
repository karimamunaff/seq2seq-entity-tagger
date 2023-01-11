from transformers import AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from paths import MODELS_DIRECTORY
from prepare_model_inputs import get_model_inputs
from transformers import DataCollatorForSeq2Seq
from config import config


def train() -> None:
    model_inputs = get_model_inputs(
        training_data_directory=MODELS_DIRECTORY,
        model_name=config["model_name"],
        input_sentence_prefix=config["input_sentence_prefix"],
        training_files_prefix=config["training_filenames_prefix"],
        training_files_extention=config["training_filenames_extention"],
    )
    model_inputs = model_inputs.prepare()

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=model_inputs.tokenizer,
        model=config["model_name"],
        padding=config["pad_batch"],
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=MODELS_DIRECTORY,
        evaluation_strategy=config["evaluation_strategy"],
        learning_rate=float(config["learning_rate"]),
        per_device_train_batch_size=int(config["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(config["per_device_eval_batch_size"]),
        weight_decay=float(config["weight_decay"]),
        save_total_limit=int(config["save_total_limit"]),
        num_train_epochs=int(config["num_train_epochs"]),
        use_mps_device=bool(int(config["use_mps_device"])),
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(model_inputs.model_name)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=model_inputs["train"],
        eval_dataset=model_inputs["test"],
        tokenizer=model_inputs.tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    train()
