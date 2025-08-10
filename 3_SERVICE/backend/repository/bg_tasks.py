"""Репозиторий для работы с фоновыми задачами в базе данных."""

import datetime as dt
from uuid import UUID

from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession

from db_models import BgTask
from serializers import BGTaskSchema


class BgTasksRepository:
    """Репозиторий для управления фоновыми задачами.
    
    Предоставляет методы для:
    - Отслеживания статуса выполнения задач
    - Создания новых задач
    - Обновления результатов выполнения
    - Удаления завершенных задач
    """
    def __init__(self, db_session: AsyncSession) -> None:
        """Инициализация репозитория с сессией БД.
        
        Args:
            db_session: Асинхронная сессия SQLAlchemy
        """
        self.db_session = db_session

    async def get_tasks(self) -> list[BgTask]:
        """Получение всех фоновых задач с сортировкой по дате обновления.
        
        Returns:
            Список задач, отсортированный по убыванию updated_at
        """
        async with self.db_session as session:
            # Получаем все задачи, отсортированные по времени обновления
            tasks: list[BgTask] = list((
                await session.execute(
                    select(
                        BgTask
                    ).order_by(BgTask.updated_at.desc())
                )
            ).scalars().all())
        return tasks

    async def create_task(self, task: BGTaskSchema) -> UUID:
        """Создание новой фоновой задачи.
        
        Args:
            task: Схема данных для создания задачи
            
        Returns:
            UUID созданной задачи
        """
        # Создаем объект задачи с автоматически генерируемым UUID
        db_bg_task = BgTask(name=task.name)
        async with self.db_session as session:
            session.add(db_bg_task)
            await session.commit()
            # в SQLAlchemy - это метод для принудительной отправки изменений в базу данных в рамках текущей транзакции, но без её завершения. - транзакция остается открытой
            await session.flush()
            return db_bg_task.uuid

    async def update_task(
        self,
        task_uuid: UUID,
        status: str,
        result_msg: str,
        updated_at: dt.datetime
    ) -> None:
        """Обновление статуса и результата выполнения задачи.
        
        Args:
            task_uuid: Уникальный идентификатор задачи
            status: Новый статус задачи
            result_msg: Сообщение с результатом выполнения
            updated_at: Время обновления
        """
        async with self.db_session as session:
            await session.execute(
                update(
                    BgTask
                ).where(
                    BgTask.uuid == task_uuid
                ).values(
                    status=status,
                    result_msg=result_msg,
                    updated_at=updated_at
                )
            )
            await session.commit()
            # в SQLAlchemy - это метод для принудительной отправки изменений в базу данных в рамках текущей транзакции, но без её завершения. - транзакция остается открытой
            await session.flush()

    async def delete_tasks_by_uuid(self, task_uuids: list[UUID]) -> None:
        """Удаление задач по списку UUID.
        
        Используется для очистки завершенных задач.
        
        Args:
            task_uuids: Список UUID задач для удаления
        """
        async with self.db_session as session:
            await session.execute(
                delete(
                    BgTask
                ).where(
                    BgTask.uuid.in_(task_uuids)
                )
            )
            await session.commit()
            # в SQLAlchemy - это метод для принудительной отправки изменений в базу данных в рамках текущей транзакции, но без её завершения. - транзакция остается открытой
            await session.flush()
