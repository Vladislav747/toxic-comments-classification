# ===== API РОУТЕР ДЛЯ МОНИТОРИНГА ФОНОВЫХ ЗАДАЧ =====
# Простой роутер для отслеживания статуса фоновых процессов обучения моделей

from typing import Annotated

from fastapi import Depends, APIRouter

from dependency import get_bg_tasks_service
from serializers import BGTaskSchema
from services import BGTasksService

# Роутер для endpoints связанных с фоновыми задачами
router = APIRouter()


@router.get(
    "/",
    response_model=list[BGTaskSchema],
    description="Получение списка задач"
)
async def get_tasks(
    bg_tasks_service: Annotated[BGTasksService, Depends(get_bg_tasks_service)]
):
    """
    Возвращает список всех фоновых задач обучения моделей.
    
    Показывает:
    - ID задачи
    - Статус (running/completed/failed)
    - Время начала
    - Время завершения (если завершена)
    - Результат или ошибку
    
    Используется для мониторинга процесса обучения моделей.
    """
    return await bg_tasks_service.get_tasks()
