from transformers import AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from paths import MODELS_DIRECTORY
from prepare_model_inputs import get_model_inputs
from transformers import DataCollatorForSeq2Seq
from config import PAD_BATCH


model_inputs = get_model_inputs()
dataset = model_inputs.prepare()
model = AutoModelForSeq2SeqLM.from_pretrained(model_inputs.model_name)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=model_inputs.tokenizer, model=model_inputs.model_name, padding=PAD_BATCH
)

training_args = Seq2SeqTrainingArguments(
    output_dir=MODELS_DIRECTORY,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    use_mps_device=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=model_inputs.tokenizer,
    data_collator=data_collator,
)
trainer.train()
trainer.save_model(MODELS_DIRECTORY)
