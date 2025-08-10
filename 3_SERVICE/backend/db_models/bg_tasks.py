"""
Модель BgTask для отслеживания фоновых задач.

Этот модуль содержит SQLAlchemy модель для хранения информации о фоновых задачах,
таких как тренировка моделей машинного обучения.
"""

import datetime as dt
from typing import Optional
from uuid import UUID

from sqlalchemy import DateTime, Enum, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from database import Base
from serializers import BGTaskStatus


class BgTask(Base):
    """
    Модель для отслеживания фоновых задач.
    
    Attributes:
        uuid: Уникальный идентификатор задачи (первичный ключ)
        name: Название задачи
        status: Статус выполнения задачи (running, completed, failed)
        result_msg: Сообщение с результатом выполнения или ошибкой
        updated_at: Время последнего обновления статуса
    """
    __tablename__ = 'bg_tasks'

    # Уникальный идентификатор задачи
    # mapped_column - это функция SQLAlchemy 2.0+ для определения колонок таблицы базы данных с поддержкой типизации. Она заменяет старый подход с Column()
    uuid: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid()
    )
    
    # Название фоновой задачи
    name: Mapped[str]
    
    # Статус выполнения задачи
    status: Mapped[str] = mapped_column(
        Enum(BGTaskStatus),
        default=BGTaskStatus.running
    )
    
    # Сообщение с результатом выполнения или ошибкой
    result_msg: Mapped[Optional[str]] = mapped_column(default=None)
    
    # Время последнего обновления
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: dt.datetime.now(dt.UTC)
    )
