# ===== PYDANTIC СХЕМЫ ДЛЯ API МОДЕЛЕЙ МЛ =====
# Этот модуль содержит все основные Pydantic схемы для:
# - Определения типов ML моделей и векторайзеров
# - Конфигурации обучения моделей
# - Запросов и ответов API endpoints
# - Валидации входных данных
# - Сериализации моделей для базы данных

import re
from enum import Enum
from pathlib import Path
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


# ===== ОБЩИЕ ОТВЕТЫ API =====

class MessageResponse(BaseModel):
    """
    Стандартный ответ API с текстовым сообщением.
    
    Используется для:
    - Подтверждений успешных операций
    - Информационных сообщений
    - Статусов выполнения операций
    
    Пример:
    {"message": "Модель успешно загружена"}
    """
    message: str  # Текст сообщения на русском языке


# ===== ПЕРЕЧИСЛЕНИЯ ТИПОВ МОДЕЛЕЙ =====

class MLModelType(str, Enum):
    """
    Перечисление доступных типов моделей машинного обучения.
    
    Классические ML модели:
    - logistic_regression: Логистическая регрессия (быстрая, интерпретируемая)
    - multinomial_naive_bayes: Наивный байесовский классификатор
    - linear_svc: Линейный SVM (хорошо для текстов)
    
    Глубокие модели:
    - distilbert: Оптимизированная версия BERT
    - deberta: DeBERTa v3 (современная трансформер модель)
    """
    # Классические модели (is_dl_model=False)
    logistic_regression = "logistic_regression"
    multinomial_nb = "multinomial_naive_bayes"
    linear_svc = "linear_svc"
    
    # Глубокие модели (is_dl_model=True)
    distilbert = "distilbert"
    deberta = "deberta"


class VectorizerType(str, Enum):
    """
    Перечисление доступных типов векторайзеров для преобразования текста.
    
    Типы векторайзеров:
    - bag_of_words (CountVectorizer): Простой подсчет слов в документе
    - tf_idf (TfidfVectorizer): Взвешивание по частоте и обратной частоте
    
    Используется только для классических ML моделей.
    Глубокие модели используют собственные tokenizer'ы.
    """
    count_vectorizer = "bag_of_words"  # Мешок слов - простой подсчет
    tfidf_vectorizer = "tf_idf"         # TF-IDF - более современный подход


# ===== КОНФИГУРАЦИЯ ОБУЧЕНИЯ МОДЕЛЕЙ =====

class MLModelConfig(BaseModel):
    """
    Конфигурация для обучения новой модели классификации токсичных комментариев.
    
    Содержит все необходимые параметры:
    - Настройки преобразования текста (векторайзер, лемматизация)
    - Тип и параметры модели машинного обучения
    - Уникальное имя модели
    
    Пример использования:
    MLModelConfig(
        name="my_toxic_classifier",
        vectorizer_type=VectorizerType.tfidf_vectorizer,
        ml_model_type=MLModelType.logistic_regression,
        vectorizer_params={"max_features": 10000, "ngram_range": (1, 2)},
        ml_model_params={"C": 1.0, "max_iter": 1000}
    )
    """
    name: str                           # Уникальное имя модели (только латинские буквы, цифры, _, -)
    spacy_lemma_tokenizer: bool = False  # Использовать ли spaCy для лемматизации (приведение к нормальной форме)
    vectorizer_type: VectorizerType     # Тип векторайзера (BOW или TF-IDF)
    vectorizer_params: dict             # Параметры векторайзера (max_features, ngram_range, и т.д.)
    ml_model_type: MLModelType          # Тип ML модели (LR, SVM, NB, BERT, и т.д.)
    ml_model_params: dict               # Параметры модели (C, max_iter, learning_rate, и т.д.)

    @field_validator("name")
    def validate_name(cls, v):
        """
        Валидатор для проверки корректности имени модели - свойства name.
        
        Правила для имени:
        - Только строчные латинские буквы (a-z)
        - Цифры (0-9)
        - Знаки подчеркивания (_) и дефисы (-)
        - Не может начинаться или заканчиваться на - или _
        
        Примеры корректных имен:
        - "my_model", "toxic_classifier_v2", "bert-fine-tuned"
        
        Некорректные имена:
        - "MyModel" (заглавные буквы), "model with spaces", "_model" (начинается с _)
        """
        if not bool(re.compile(r"^[a-z0-9_]+(?:[-_][a-z0-9_]+)*$").match(v)):
            raise ValueError(
                "Имя модели может состоять только из строчных латинских букв, "
                "цифр,  дефисов, и знаков подчеркивания"
            )
        return v


# ===== ЗАПРОСЫ УПРАВЛЕНИЯ МОДЕЛЯМИ =====

class LoadRequest(BaseModel):
    """
    Запрос на загрузку модели в оперативную память.
    
    Используется для endpoint POST /api/v1/models/load
    После успешной загрузки модель доступна для предсказаний.
    """
    name: str  # Имя модели для загрузки в память


class UnloadRequest(LoadRequest):
    """
    Запрос на выгрузку модели из оперативной памяти.
    
    Наследует от LoadRequest - имеет ту же структуру.
    Используется для endpoint POST /api/v1/models/unload
    Освобождает место в памяти для других моделей.
    """
    pass  # Полностью наследует структуру LoadRequest


# ===== ЗАПРОСЫ И ОТВЕТЫ ПРЕДСКАЗАНИЙ =====

class PredictRequest(BaseModel):
    """
    Запрос на предсказание токсичности текстов.
    
    Может содержать один или несколько текстов для анализа.
    Используется в endpoint POST /api/v1/models/predict/{model_name}
    
    Пример:
    {
        "X": [
            "Привет, как дела?",
            "Ты дурак и идиот!"
        ]
    }
    
    Важно: кавычки в тексте необходимо экранировать.
    """
    X: list[str] = Field(
        description="Кавычки в тексте необходимо экранировать"
    )  # Список текстов для классификации


class PredictResponse(BaseModel):
    """
    Ответ с результатами предсказания токсичности.
    
    Содержит бинарные предсказания для каждого текста.
    Порядок соответствует порядку в запросе.
    
    Пример ответа:
    {
        "predictions": [0, 1]  # 0 = нетоксичный, 1 = токсичный
    }
    """
    predictions: list[int]  # Список бинарных предсказаний: 0 (нетоксичный) или 1 (токсичный)


# ===== СХЕМЫ МОДЕЛЕЙ ДЛЯ БАЗЫ ДАННЫХ =====

class MLModelBaseSchema(BaseModel):
    """
    Базовая схема модели машинного обучения.
    
    Содержит общие поля для всех моделей:
    - Метаданные (имя, тип)
    - Статусы (обучена, загружена)
    - Параметры конфигурации
    
    Используется как родительский класс для специализированных схем.
    """
    name: str                                    # Уникальное имя модели
    type: MLModelType                           # Тип ML модели (из enum)
    is_trained: bool = False                    # Завершено ли обучение (есть ли файл модели)
    is_loaded: bool = False                     # Загружена ли в оперативную память (loaded_models)
    model_params: dict = Field(default_factory=dict)      # Параметры ML модели (C, max_iter, и т.д.)
    vectorizer_params: dict = Field(default_factory=dict) # Параметры векторайзера (max_features, ngram_range, и т.д.)


class MLModelCreateSchema(MLModelBaseSchema):
    """
    Схема для создания новой модели в базе данных.
    
    Расширяет базовую схему дополнительными полями:
    - Признак глубокой модели
    - Путь к файлу модели
    
    Используется в ORM операциях создания записей.
    """
    is_dl_model: bool = False                   # True для глубоких моделей (BERT, DeBERTa), False для классических (LR, SVM)
    saved_model_file_path: Path | None = None   # Путь к сохраненному файлу модели (после обучения)


class MLModelInListResponse(MLModelBaseSchema):
    """
    Схема модели для отображения в списке.
    
    Используется в endpoint GET /api/v1/models/ для возврата
    списка всех доступных моделей.
    
    Отличие от базовой схемы:
    - Добавлен UUID для уникальной идентификации
    - Настроен для автоматического преобразования из ORM моделей
    """
    uuid: UUID  # Уникальный идентификатор модели в базе данных

    class Config:
        from_attributes = True  # Позволяет создавать Pydantic модель из SQLAlchemy ORM объекта
