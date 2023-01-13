from transformers import AutoModelForSeq2SeqLM
from src.paths import MODELS_DIRECTORY
from transformers import AutoTokenizer
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained(MODELS_DIRECTORY)
model = AutoModelForSeq2SeqLM.from_pretrained(MODELS_DIRECTORY)
translator = pipeline("translation", model=MODELS_DIRECTORY)


def get_predictions(example):
    example_prefix = f"Tag Entities: {example}"
    inputs = tokenizer(example_prefix, return_tensors="pt").input_ids
    outputs = model.generate(
        inputs, max_new_tokens=40, do_sample=True, top_k=20, top_p=0.8
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


print(get_predictions("Tag Entities: Capitalism vs communism in the news"))
