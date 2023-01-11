from transformers import AutoModelForSeq2SeqLM
from paths import MODELS_DIRECTORY
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("t5-small")

model = AutoModelForSeq2SeqLM.from_pretrained(MODELS_DIRECTORY).to("mps").eval()


def get_predictions(example):
    example_prefix = f"Tag Entities: {example}"
    example_tokenized = tokenizer(
        example_prefix, text_target=example, return_tensors="pt"
    ).to("mps")
    outputs = model(**example_tokenized)
    predictions = outputs.logits.argmax(-1)
    return tokenizer.decode(predictions[0])


a = 1
