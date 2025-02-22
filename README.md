# Fine-tuning LLaMA 3 for Question Answering

Этот проект представляет собой настройку и дообучение модели **LLaMA 3 (8B)** для задачи ответа на вопросы. Используется **QLoRA** для эффективного обучения с низким потреблением памяти.

## 📌 Особенности
- Использование **LLaMA 3 (8B)** для генерации ответов
- **QLoRA** (4-bit quantization) для экономии памяти
- Обучение и тестирование на предоставленном датасете
- Вывод результатов в консоль (без сохранения в файл)

## 📂 Структура проекта
- `main.py` — основной код для загрузки данных, обучения и тестирования модели
- `LR1.csv` — основной датасет для обучения (300 вопросов)
- `LR1_dev.csv` — тестовый набор (100 вопросов)
- `LR1_dev_answer.csv` — правильные ответы для тестов

## 🚀 Установка
### 1️⃣ Установка зависимостей
Убедитесь, что у вас установлен Python **3.8+** и выполните:
```bash
pip install -r requirements.txt
```

### 2️⃣ Настройка окружения
Перед запуском убедитесь, что у вас есть **CUDA** и установлен PyTorch с GPU-поддержкой:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
Если CUDA недоступна, переустановите PyTorch:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 3️⃣ Укажите ваш **Hugging Face Token** в `main.py`
```python
HF_TOKEN = "your_huggingface_token"
```

## 📊 Запуск обучения
Запустите основной скрипт:
```bash
python main.py
```
Будут выведены этапы работы скрипта, а также результаты предсказаний модели.

## 🔬 Формат входных данных
Каждая строка содержит:
- **Вопрос** (question)
- **Категория** (subject)
- **4 варианта ответа** (choices)

Пример:
```
Which among the following would result in the narrowest confidence interval?
['Small sample size and 95% confidence',
 'Small sample size and 99% confidence',
 'Large sample size and 95% confidence',
 'Large sample size and 99% confidence']
✅ Правильный ответ: 2
```

## 🛠 Тестирование модели
После обучения модель тестируется на `LR1_dev.csv`. Результаты предсказаний выводятся в консоль в формате:
```
Вопрос: What is the capital of France?
Ответ: Paris
--------------------------------------------------
```

## 📌 Заметки
- Поддерживаются только модели **llama3-8b-8192** и **llama3-70b-8192**
- Файл с ответами не создается, а выводится в терминал
- В будущем можно расширить поддержку других LLaMA-моделей

## 📜 Лицензия
Проект предназначен для образовательных целей. Использование модели должно соответствовать условиям **Hugging Face** и **Meta AI**.

---
💡 **Разработано для соревновательного тестирования LLaMA 3.**

