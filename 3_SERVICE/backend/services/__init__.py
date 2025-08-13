"""
Слой бизнес-логики (Service Layer) для обработки операций приложения.

Содержит сервисы для:
- Управления фоновыми задачами (BGTasksService)
- Обучения и управления ML моделями (TrainerService)

Сервисы реализуют бизнес-логику и координируют работу между контроллерами и репозиториями.
"""

from services.background_tasks import BGTasksService
from services.trainer import TrainerService

__all__ = ['BGTasksService', 'TrainerService']
