"""
Сервис для управления фоновыми задачами.

Обеспечивает мониторинг и управление жизненным циклом асинхронных задач.
"""

from repository import BgTasksRepository
from serializers import BGTaskSchema
from settings import app_config


class BGTasksService:
    """
    Сервис для мониторинга и управления фоновыми задачами.
    
    Основные функции:
    - Получение списка всех задач
    - Автоматическая очистка старых завершенных задач
    - Поддержание оптимального размера истории задач
    """

    def __init__(
        self,
        bg_tasks_repo: BgTasksRepository,
    ):
        """
        Инициализация сервиса с репозиторием для доступа к данным.
        
        Args:
            bg_tasks_repo: Репозиторий для работы с фоновыми задачами в БД
        """
        self.bg_tasks_repo = bg_tasks_repo

    async def get_tasks(self) -> list[BGTaskSchema]:
        """
        Получение списка всех фоновых задач.
        
        Преобразует данные из БД в сериализованные объекты для API.
        
        Returns:
            Список сериализованных задач, отсортированных по времени обновления
        """
        # Получаем сырые данные из БД через репозиторий
        tasks = await self.bg_tasks_repo.get_tasks()
        # Преобразуем каждую задачу в сериализованный формат
        return [BGTaskSchema.model_validate(task) for task in tasks]

    async def rotate_tasks(self) -> None:
        """
        Автоматическая ротация (очистка) старых фоновых задач.
        
        Удаляет самые старые завершенные задачи, если общее количество
        превышает максимально допустимое значение.
        
        Логика:
        1. Получаем все задачи
        2. Проверяем превышение лимита
        3. Фильтруем только завершенные задачи
        4. Удаляем лишние (самые старые)
        """
        # Получаем все задачи для проверки количества
        bg_tasks = await self.bg_tasks_repo.get_tasks()
        
        # Проверяем, нужна ли очистка - превысили ли мы лимит из конфига
        if len(bg_tasks) > app_config.max_saved_bg_tasks:
            # Вычисляем количество лишних задач
            excess_count = len(bg_tasks) - app_config.max_saved_bg_tasks
            
            # Отбираем только завершенные задачи (статус success или failure)
            # Берем последние ([-excess_count:]) - это самые старые
            task_ids_to_remove = [
                task.uuid for task in bg_tasks
                if task.status in ("success", "failure")  # Только завершенные задачи
            ][-excess_count:]  # Самые старые из лишних

            # Выполняем удаление через репозиторий
            await self.bg_tasks_repo.delete_tasks_by_uuid(task_ids_to_remove)
