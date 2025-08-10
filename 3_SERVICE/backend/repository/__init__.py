"""
Repository layer для работы с базой данных.

Содержит репозитории для:
- Управления ML моделями (ModelsRepository)
- Управления фоновыми задачами (BgTasksRepository)

Репозитории реализуют паттерн Repository для абстракции работы с БД.
"""

from repository.bg_tasks import BgTasksRepository
from repository.models import ModelsRepository


__all__ = ['ModelsRepository', 'BgTasksRepository']
