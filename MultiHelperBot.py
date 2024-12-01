import asyncio
import logging
import re

from aiogram import Bot, Dispatcher, Router, types
from aiogram.filters import Command
from aiogram.utils.keyboard import ReplyKeyboardBuilder
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
import numpy as np
from nltk.tokenize import RegexpTokenizer

# Загрузка необходимых данных NLTK
nltk.download('punkt')

# ======== Конфигурация бота ========

API_TOKEN = '##############'  # Замените на ваш токен

# ======== Данные для обучения модели ========

training_data = [
    ("как перевести hello", "Привет."),
    ("как перевести hi", "Привет."),
    ("как сказать пока по-английски", "Bye."),
    ("что значит cat", "Кошка."),
    ("как перевести good night", "Спокойной ночи."),
    ("что значит good morning", "Доброе утро."),
    ("как переводится thank you", "Спасибо."),
    ("что значит please", "Пожалуйста."),
    ("как перевести bye", "Пока."),
    ("что такое hello", "Привет."),
    ("что такое good night", "Спокойной ночи."),
    ("как переводится good night", "Спокойной ночи."),
    ("что значит dog", "Собака."),
    ("как перевести слово dog", "Собака."),
    ("как перевести слово cat", "Кошка."),
    ("как перевести good morning", "Доброе утро."),
    ("что значит fox", "Лиса."),
    ("как перевести слово fox", "Лиса."),
]

# ======== Предобработка текста ========

def preprocess_text(text):
    """Предварительная обработка текста: приведение к нижнему регистру, удаление пунктуации и лишних пробелов."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Удаление пунктуации
    text = re.sub(r'\s+', ' ', text).strip()  # Удаление лишних пробелов
    return text

# ======== Обучение модели ========

# Подготовка данных для обучения
X_train_raw = [preprocess_text(text) for text, _ in training_data]
y_train = [response for _, response in training_data]

# Векторизация запросов
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train_raw)

print("Модель успешно подготовлена!")

# ======== Функция предсказания ========

def predict_response(user_input):
    """Предсказание ответа на основе косинусного сходства с тренировочными данными."""
    user_input_processed = preprocess_text(user_input)
    logging.info(f"Обработанный запрос: {user_input_processed}")

    # Токенизация входного сообщения
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(user_input_processed)
    logging.info(f"Токены запроса: {tokens}")

    # Добавление чисел в токены (если есть)
    tokens_with_numbers = []
    for token in tokens:
        if any(char.isdigit() for char in token):
            tokens_with_numbers.append(token)
    if tokens_with_numbers:
        logging.info(f"Токены с числами: {tokens_with_numbers}")

    # Получение бинарного представления токенов
    binary_tokens = [' '.join(format(ord(char), '08b') for char in token) for token in tokens]
    logging.info(f"Бинарное представление токенов:")
    for token, binary in zip(tokens, binary_tokens):
        logging.info(f"Токен: {token}, Бинарное представление: {binary}")

    try:
        user_input_vec = vectorizer.transform([user_input_processed])

        # Вычисляем косинусное сходство
        similarities = np.dot(X_train_vec, user_input_vec.T).toarray().flatten()
        logging.debug(f"Сходства с тренировочными данными: {similarities}")

        # Находим индекс наиболее похожего запроса
        best_match_idx = similarities.argmax()
        best_match_similarity = similarities[best_match_idx]

        logging.info(f"Наибольшее сходство: {best_match_similarity}")

        # Если сходство достаточно высокое, возвращаем ответ
        if best_match_similarity > 0.3:  # Порог можно настроить
            response = y_train[best_match_idx]
            logging.info(f"Выбранный ответ: {response}")
            return response
        else:
            return "Извините, я пока не знаю, как ответить на этот вопрос."
    except Exception as e:
        logging.error(f"Ошибка предсказания: {e}")
        return "Извините, я пока не знаю, как ответить на этот вопрос."

# ======== Настройка бота ========

# Инициализация логирования
logging.basicConfig(level=logging.INFO)

# Инициализация бота и диспетчера
bot = Bot(token=API_TOKEN)
dp = Dispatcher()
router = Router()

@router.message(Command("start", "help"))
async def cmd_start(message: types.Message):
    """Обработчик команд /start и /help."""
    await message.answer(
        "Привет! Я бот-помощник. Вот что я умею:\n"
        "1. Помочь с тренировками (напишите 'тренировка').\n"
        "2. Ответить на университетские вопросы (напишите 'университет').\n"
        "3. Помочь с языковыми вопросами (например, 'Как перевести hello?')."
    )

# ======== Сценарий 1: Фитнес-тренировки ========

@router.message(lambda msg: "тренировка" in msg.text.lower())
async def fitness_handler(message: types.Message):
    """Обработчик сценария фитнес-тренировок."""
    builder = ReplyKeyboardBuilder()
    builder.add(types.KeyboardButton(text="Новичок"))
    builder.add(types.KeyboardButton(text="Средний"))
    builder.add(types.KeyboardButton(text="Продвинутый"))
    builder.adjust(1)
    await message.answer(
        "Выберите ваш уровень подготовки:",
        reply_markup=builder.as_markup(resize_keyboard=True)
    )

@router.message(lambda msg: msg.text in ["Новичок", "Средний", "Продвинутый"])
async def fitness_plan(message: types.Message):
    """Предоставляет план тренировки на основе уровня пользователя."""
    plans = {
        "Новичок": "План для новичка: Разминка, 10 минут кардио, растяжка.",
        "Средний": "План для среднего уровня: 20 минут кардио, силовые упражнения.",
        "Продвинутый": "План для продвинутого уровня: Интервальные тренировки, бег 5 км, йога."
    }
    response = plans.get(message.text, "Извините, я не знаю такого уровня.")
    await message.answer(response, reply_markup=types.ReplyKeyboardRemove())

# ======== Сценарий 2: Университетский консультант ========

@router.message(lambda msg: "университет" in msg.text.lower())
async def university_handler(message: types.Message):
    """Обработчик сценария университетского консультанта."""
    builder = ReplyKeyboardBuilder()
    builder.add(types.KeyboardButton(text="Расписание"))
    builder.add(types.KeyboardButton(text="Сессия"))
    builder.add(types.KeyboardButton(text="Контакты"))
    builder.adjust(1)
    await message.answer(
        "Какой тип информации вас интересует?",
        reply_markup=builder.as_markup(resize_keyboard=True)
    )

@router.message(lambda msg: msg.text in ["Расписание", "Сессия", "Контакты"])
async def university_info(message: types.Message):
    """Предоставляет информацию об университете на основе выбора пользователя."""
    responses = {
        "Расписание": "Расписание занятий доступно на официальном сайте университета.",
        "Сессия": "Сессия начинается 15 января. Удачи в подготовке!",
        "Контакты": "Контакты деканата: тел. 123-456-789, email: dean@example.com."
    }
    response = responses.get(message.text, "Извините, я не могу предоставить эту информацию.")
    await message.answer(response, reply_markup=types.ReplyKeyboardRemove())

# ======== Сценарий 3: Языковой помощник ========

@router.message(lambda msg: True)
async def language_handler(message: types.Message):
    """Обработчик языковых запросов."""
    user_input = message.text
    response = predict_response(user_input)
    await message.answer(response)

# ======== Запуск бота ========

async def main():
    """Основная функция для запуска бота."""
    dp.include_router(router)

    # Удаление предыдущих вебхуков и запуск поллинга
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
