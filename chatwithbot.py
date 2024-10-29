import re
from transformers import Trainer, TrainingArguments, GPT2LMHeadModel, GPT2Tokenizer
from datasets import Dataset
import torch

class ChatBot:
    def __init__(self, model_path="./chatbot_model"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def get_response(self, user_input, max_length=150):
        inputs = self.tokenizer(user_input, return_tensors="pt", padding=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.pad_token_id  # Padding token'ını doğru ayarla
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def start_chat(self):
        while True:
            user_input = input("Sen: ")
            if user_input.lower() in ["quit", "exit", "q"]:break
            response = self.get_response(user_input)
            print(f"Bot: {response}")

if __name__ == "__main__":
    chatbot = ChatBot()
    chatbot.start_chat()
