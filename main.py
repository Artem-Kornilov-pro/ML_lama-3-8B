import pandas as pd
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from trl import SFTTrainer
import torch

print("Проверка доступности CUDA...")
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

# Настройки модели
MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
HF_TOKEN = "your_huggingface_token"  # Укажите ваш токен

print("Загрузка данных...")
# Загрузка данных
def load_data():
    """Загружает CSV-файлы с данными для обучения и тестирования."""
    df = pd.read_csv("LR1.csv")
    df_test = pd.read_csv("LR1_dev.csv")
    df_test_answers = pd.read_csv("LR1_dev_answer.csv")
    return df, df_test, df_test_answers

df, df_test, df_test_answers = load_data()

print("Настройка токенизатора...")
# Настройка токенизатора
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

print("Настройка QLoRA...")
# QLoRA конфигурация
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("Загрузка модели...")
# Загрузка модели
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, quantization_config=bnb_config, device_map="auto"
)

print("Настройка LoRA...")
# Конфигурация LoRA
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj']
)

print("Подготовка модели к обучению...")
# Подготовка модели к обучению
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

print("Форматирование данных для обучения...")
# Форматирование данных для обучения
def format_example(row):
    """Форматирует строку датасета в нужный формат."""
    return {
        "input_text": f"{row['question']}\n{row['choices']}",
        "label": row["answer"]
    }

dataset = Dataset.from_pandas(df).map(format_example)

dataset = dataset.train_test_split(test_size=0.1)

print("Настройка параметров обучения...")
# Настройки обучения
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    save_steps=500,
    logging_steps=50,
    learning_rate=2e-4,
    save_total_limit=2,
    evaluation_strategy="epoch",
    push_to_hub=False
)

print("Создание тренера и запуск обучения...")
# Обучение
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    args=training_args,
    tokenizer=tokenizer,
)

trainer.train()

print("Тестирование модели...")
# Тестирование модели
def evaluate_model():
    """Запускает тестирование модели и выводит предсказания в консоль."""
    predictions = []
    for _, row in df_test.iterrows():
        input_text = f"{row['question']}\n{row['choices']}"
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
        output = model.generate(**inputs, max_length=50)
        pred = tokenizer.decode(output[0], skip_special_tokens=True)
        predictions.append(pred)
        print(f"Вопрос: {row['question']}")
        print(f"Ответ: {pred}")
        print("-" * 50)

# Запуск тестирования
evaluate_model()
print("Готово!")
