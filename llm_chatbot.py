import re
from transformers import Trainer, TrainingArguments, GPT2LMHeadModel, GPT2Tokenizer
from datasets import Dataset
import torch

# 1. Veriyi Yükleme ve Filtreleme
def load_and_filter_data(file_path, speaker="---"):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # mesajlarını filtreleme
    filtered_messages = []
    for line in lines:
        if f"{speaker}:" in line:
            # Mesajı ayıkla ve temizle
            message = re.sub(r".*?: ", "", line).strip()  # Tarih ve isimleri kaldır
            filtered_messages.append(message)

    return filtered_messages

# 2. Veriyi dataset olarak formatlama
def prepare_dataset(filtered_messages):
    # Her mesajı bir satır olarak değerlendireceğiz.
    return Dataset.from_dict({"text": filtered_messages})

# 3. Modeli Eğitme
def train_model(dataset):
    # Tokenizer ve Modeli yükleme
    model = GPT2LMHeadModel.from_pretrained("gpt2/")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2/")

    # Padding token'ı ayarlama
    tokenizer.pad_token = tokenizer.eos_token  # `eos_token`'ı padding olarak kullanabilirsiniz

    # Veriyi tokenlara dönüştürme
    def tokenize_function(examples):
        tokenized_inputs = tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)
        tokenized_inputs['labels'] = tokenized_inputs['input_ids'].copy()  # Input_ids'ı etiket olarak ayarla
        return tokenized_inputs

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Eğitim argümanları
    training_args = TrainingArguments(
        output_dir="./chatbot_model",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=500,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=100,
    )

    # Model Eğitme
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    trainer.train()

    # Eğitilen modeli kaydetme
    trainer.save_model("./chatbot_model")
    tokenizer.save_pretrained("./chatbot_model")

# 4. Chatbot ile Muhabbet
def chat_with_bot():
    tokenizer = GPT2Tokenizer.from_pretrained("./chatbot_model")
    model = GPT2LMHeadModel.from_pretrained("./chatbot_model")

    while True:
        user_input = input("Sen: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            break

        # Kullanıcıdan gelen girdiyi modele verip, yanıt alalım
        inputs = tokenizer(user_input, return_tensors="pt")
        outputs = model.generate(inputs.input_ids, max_length=150, num_return_sequences=1)

        # Modelin cevabını yazdırma
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Bot: {response}")

# Ana Akış
if __name__ == "__main__":
    # Veri dosyasının yolu
    file_path = "----.txt"

    #  konuşmalarını filtrele
    filtered_messages = load_and_filter_data(file_path, speaker="----")

    # Dataset hazırlama
    dataset = prepare_dataset(filtered_messages)

    # Model eğitme
    train_model(dataset)

    # Chatbot ile konuşma
    chat_with_bot()
