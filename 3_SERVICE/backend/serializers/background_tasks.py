# ===== PYDANTIC СХЕМЫ ДЛЯ ФОНОВЫХ ЗАДАЧ =====
# Схемы для отслеживания статуса и результатов асинхронных процессов обучения моделей

from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class BGTaskStatus(str, Enum):
    """
    Перечисление возможных статусов фоновых задач обучения моделей.
    
    Статусы:
    - running: Задача выполняется (процесс обучения активен)
    - success: Задача завершена успешно (модель обучена и сохранена)  
    - failure: Задача завершена с ошибкой (обучение прервано)
    
    Используется для мониторинга прогресса через API /api/v1/tasks/
    """
    running = "running"   # Задача в процессе выполнения
    success = "success"   # Задача успешно завершена
    failure = "failure"   # Задача завершена с ошибкой


class BGTaskSchema(BaseModel):
    """
    Схема фоновой задачи для отслеживания процесса обучения моделей.
    
    Каждая задача обучения модели (POST /api/v1/models/fit) создает
    запись в базе данных для мониторинга прогресса.
    
    Поля:
    - uuid: Уникальный идентификатор задачи
    - name: Имя обучаемой модели  
    - status: Текущий статус выполнения
    - result_msg: Сообщение о результате (успех/ошибка)
    - updated_at: Время последнего обновления статуса
    
    Пример использования:
    {
        "uuid": "123e4567-e89b-12d3-a456-426614174000",
        "name": "my_toxic_classifier",
        "status": "running",
        "result_msg": null,
        "updated_at": "2024-01-15T10:30:00"
    }
    """
    uuid: UUID = Field(default_factory=uuid4)      # Уникальный ID задачи (генерируется автоматически)
    name: str                                       # Имя модели которая обучается
    status: BGTaskStatus = BGTaskStatus.running     # Текущий статус (по умолчанию - running)
    result_msg: str | None = None                   # Сообщение о результате выполнения (заполняется при завершении)
    updated_at: datetime | None = datetime.now()   # Время последнего обновления статуса

    class Config:
        from_attributes = True  # Позволяет создавать Pydantic модель из SQLAlchemy ORM объекта
