import re
from transformers import Trainer, TrainingArguments, GPT2LMHeadModel, GPT2Tokenizer
from datasets import Dataset
import torch


def load_and_filter_data(file_path, speaker="--"):
    with open(file_path, 'r', encoding='utf-8') as file: lines = file.readlines()

    filtered_messages = []
    for line in lines:
        if f"{speaker}:" in line:
            message = re.sub(r".*?: ", "", line).strip()
            filtered_messages.append(message)

    return filtered_messages

def prepare_dataset(filtered_messages): return Dataset.from_dict({"text": filtered_messages})


def train_model(dataset):
    model = GPT2LMHeadModel.from_pretrained("---")
    tokenizer = GPT2Tokenizer.from_pretrained("---")

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    def tokenize_function(examples):
        tokenized_inputs = tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)
        tokenized_inputs['labels'] = tokenized_inputs['input_ids'].copy()
        return tokenized_inputs

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir="./chatbot_model",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=500,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=100,
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset,)

    trainer.train()

    trainer.save_model("./chatbot_model")
    tokenizer.save_pretrained("./chatbot_model")

if __name__ == "__main__":
    file_path = "---.txt"
    filtered_messages = load_and_filter_data(file_path, speaker="---")
    dataset = prepare_dataset(filtered_messages)
    train_model(dataset)

