"""
Модель Model для хранения информации о ML моделях.

Этот модуль содержит SQLAlchemy модель для хранения метаданных о моделях машинного обучения,
включая их параметры, состояние тренировки и загрузки.
"""

from uuid import UUID

from sqlalchemy import Enum, JSON, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from database import Base
from serializers import MLModelType


class Model(Base):
    """
    Модель для хранения информации о ML моделях.
    
    Attributes:
        uuid: Уникальный идентификатор модели (первичный ключ)
        name: Уникальное название модели
        type: Тип модели (LogisticRegression, SVC, MultinomialNB и т.д.)
        is_dl_model: Флаг, указывающий является ли модель глубокого обучения
        is_trained: Флаг, указывающий обучена ли модель
        is_loaded: Флаг, указывающий загружена ли модель в память
        model_params: JSON с параметрами модели
        vectorizer_params: JSON с параметрами векторизатора
        saved_model_file_path: Путь к сохраненному файлу модели
    """
    __tablename__ = 'models'

    # Уникальный идентификатор модели
    # mapped_column - это функция SQLAlchemy 2.0+ для определения колонок таблицы базы данных с поддержкой типизации. Она заменяет старый подход с Column()
    uuid: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid()
    )
    
    # Уникальное название модели с индексом для быстрого поиска
    name: Mapped[str] = mapped_column(unique=True, nullable=False, index=True)
    
    # Тип модели машинного обучения
    type: Mapped[MLModelType] = mapped_column(
        Enum(MLModelType),
        nullable=False
    )
    
    # Флаг модели глубокого обучения
    is_dl_model: Mapped[bool] = mapped_column(default=False)
    
    # Флаг обученности модели
    is_trained: Mapped[bool] = mapped_column(default=False)
    
    # Флаг загруженности модели в память
    is_loaded: Mapped[bool] = mapped_column(default=False)
    
    # Параметры модели в формате JSON
    model_params: Mapped[dict] = mapped_column(JSON, default=dict)
    
    # Параметры векторизатора в формате JSON
    vectorizer_params: Mapped[dict] = mapped_column(JSON, default=dict)
    
    # Путь к файлу сохраненной модели
    saved_model_file_path: Mapped[str] = mapped_column(nullable=True)
