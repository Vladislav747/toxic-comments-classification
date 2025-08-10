"""Репозиторий для работы с ML моделями в базе данных."""

from uuid import UUID

from sqlalchemy import select, delete, update
from sqlalchemy.ext.asyncio import AsyncSession

from db_models import Model
from serializers import MLModelCreateSchema
from store import DEFAULT_MODELS_INFO


class ModelsRepository:
    """Репозиторий для управления ML моделями в базе данных.
    
    Предоставляет методы для CRUD операций с моделями:
    - Создание, чтение, обновление и удаление моделей
    - Фильтрация по типу модели (DL/ML)
    - Управление статусами загрузки и обученности
    """
    def __init__(self, db_session: AsyncSession):
        """Инициализация репозитория с сессией БД.
        
        Args:
            db_session: Асинхронная сессия SQLAlchemy
        """
        self.db_session = db_session

    async def get_models(self, is_dl: bool | None = None) -> list[Model]:
        """Получение списка всех моделей с опциональной фильтрацией.
        
        Args:
            is_dl: Фильтр по типу модели (True - DL, False - ML, None - все)
            
        Returns:
            Список моделей из БД
        """
        async with self.db_session as session:
            stmt = select(Model)
            # Применяем фильтр по типу модели если указан
            if is_dl is not None:
                stmt = stmt.where(Model.is_dl_model == is_dl)
            result = await session.execute(stmt)
            models = list(result.scalars().all())
        return models

    async def get_model_by_name(self, model_name: str) -> Model | None:
        """Получение модели по имени.
        
        Args:
            model_name: Имя модели для поиска
            
        Returns:
            Модель или None если не найдена
        """
        async with self.db_session as session:
            model: Model = (
                await session.execute(
                    select(
                        Model
                    ).where(
                        Model.name == model_name
                    )
                )
            ).scalar_one_or_none()
        return model

    async def get_models_by_names(self, model_names: list[str]) -> list[Model]:
        """Получение моделей по списку имен.
        
        Args:
            model_names: Список имен моделей для поиска
            
        Returns:
            Список найденных моделей
        """
        async with self.db_session as session:
            models: list[Model] = list((
                await session.execute(
                    select(
                        Model
                    ).where(
                        Model.name.in_(model_names)
                    )
                )
            ).scalars().all())
        return models

    async def create_model(self, model: MLModelCreateSchema) -> UUID:
        """Создание новой модели в БД.
        
        Args:
            model: Схема данных для создания модели
            
        Returns:
            UUID созданной модели
        """
        db_model = Model(
            name=model.name,
            type=model.type,
            is_dl_model=model.is_dl_model,
            is_trained=model.is_trained,
            is_loaded=model.is_loaded,
            model_params=model.model_params,
            vectorizer_params=model.vectorizer_params,
            saved_model_file_path=str(model.saved_model_file_path)
        )
        async with self.db_session as session:
            session.add(db_model)
            await session.commit()
            await session.flush()
            return db_model.uuid

    async def delete_model(self, model_name: str) -> None:
        """Удаление модели по имени.
        
        Args:
            model_name: Имя модели для удаления
        """
        async with self.db_session as session:
            await session.execute(
                delete(
                    Model
                ).where(
                    Model.name == model_name
                )
            )
            await session.commit()
            await session.flush()

    async def update_model_is_loaded(
        self,
        model_name: str,
        is_loaded: bool
    ) -> None:
        """Обновление статуса загрузки модели.
        
        Args:
            model_name: Имя модели
            is_loaded: Новый статус загрузки
        """
        async with self.db_session as session:
            await session.execute(
                update(
                    Model
                ).where(
                    Model.name == model_name
                ).values(
                    is_loaded=is_loaded
                )
            )
            await session.commit()
            await session.flush()

    async def update_model_after_training(
        self,
        model_name: str,
        is_trained: bool,
        model_params: dict,
        vectorizer_params: dict,
        saved_model_file_path: str
    ) -> None:
        """Обновление модели после завершения обучения.
        
        Обновляет параметры модели, векторизатора и путь к файлу.
        
        Args:
            model_name: Имя модели
            is_trained: Статус обученности
            model_params: Параметры обученной модели
            vectorizer_params: Параметры векторизатора
            saved_model_file_path: Путь к сохраненному файлу модели
        """
        async with self.db_session as session:
            await session.execute(
                update(
                    Model
                ).where(
                    Model.name == model_name
                ).values(
                    is_trained=is_trained,
                    model_params=model_params,
                    vectorizer_params=vectorizer_params,
                    saved_model_file_path=saved_model_file_path
                )
            )
            await session.commit()
            await session.flush()

    async def delete_all_user_models(self) -> list[Model]:
        """Удаление всех пользовательских моделей.
        
        Удаляет все модели кроме моделей по умолчанию из DEFAULT_MODELS_INFO.
        
        Returns:
            Список удаленных моделей
        """
        # Получаем имена всех пользовательских моделей (исключая дефолтные)
        # 1. Асинхронно получаем все модели из БД
        all_models = await self.get_models()
        
        # 2. Фильтруем только пользовательские модели (не системные)
        # DEFAULT_MODELS_INFO содержит имена встроенных моделей, которые нельзя удалять
        
        # эквивалентно:
        # model_to_remove_names = []

        # for model in all_models:
        #     if model.name not in DEFAULT_MODELS_INFO:
        #         model_to_remove_names.append(model.name)
        model_to_remove_names = [
            model.name                              # Извлекаем имя модели
            for model in all_models                 # Итерируемся по всем моделям
            if model.name not in DEFAULT_MODELS_INFO  # Исключаем системные модели
        ]
        async with self.db_session as session:
            deleted_models: list[Model] = list((
                await session.execute(
                    delete(
                        Model
                    ).where(
                        Model.name.in_(model_to_remove_names)
                    ).returning(Model)
                )
            ).scalars().all())
            await session.commit()
            await session.flush()
            return deleted_models
