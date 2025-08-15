# ===== КОНФИГУРАЦИЯ АСИНХРОННОЙ БАЗЫ ДАННЫХ =====
# Модуль настраивает подключение к PostgreSQL через SQLAlchemy 2.0 async
# Обеспечивает управление сессиями и базовый класс для ORM моделей

from typing import Any

from sqlalchemy.orm import DeclarativeBase, declared_attr
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    create_async_engine,
    async_sessionmaker
)

from settings import app_config

# ===== АСИНХРОННЫЙ ДВИЖОК БАЗЫ ДАННЫХ =====
# Создаем подключение к PostgreSQL с оптимизированными настройками
engine = create_async_engine(
    app_config.db_url,      # URL подключения из переменных окружения (DB_URL)
    future=True,            # Включаем новый API SQLAlchemy 2.0 (современный подход)
    echo=True,              # Логируем все SQL запросы в консоль (для отладки)
    pool_pre_ping=True      # Проверяем соединения перед использованием (защита от обрывов)
)

# ===== ФАБРИКА АСИНХРОННЫХ СЕССИЙ =====
# Создает сессии для работы с БД в асинхронном режиме
AsyncSessionFactory = async_sessionmaker(
    engine,                 # Используем наш движок
    autoflush=False,        # НЕ автоматически сбрасываем изменения (контролируем вручную)
    expire_on_commit=False  # НЕ истекают объекты после коммита (можем использовать дальше)
)


async def get_db_session() -> AsyncSession:
    """
    Генератор асинхронных сессий базы данных для FastAPI Depends().
    
    Автоматически создает новую сессию для каждого запроса API,
    обеспечивает правильное закрытие соединения после завершения запроса.
    
    Использование:
    async def my_endpoint(session: AsyncSession = Depends(get_db_session)):
        # session автоматически создается и закрывается
        result = await session.execute(select(MLModel))
    
    Преимущества:
    - Автоматическое управление жизненным циклом сессии
    - Защита от утечек соединений
    - Интеграция с FastAPI dependency injection
    """
    async with AsyncSessionFactory() as session:
        yield session  # Возвращаем сессию в endpoint, после выполнения автоматически закрываем


# ===== БАЗОВЫЙ КЛАСС ДЛЯ ORM МОДЕЛЕЙ =====
class Base(DeclarativeBase):
    """
    Базовый класс для всех SQLAlchemy моделей в проекте.
    
    Автоматически предоставляет:
    - Автогенерацию имен таблиц (от имени класса)
    - Базовые поля и настройки
    - Совместимость с современным SQLAlchemy 2.0
    
    Пример использования:
    class MLModel(Base):
        __tablename__ = "ml_models"  # или автоматически: "mlmodel"
        id: Mapped[UUID] = mapped_column(primary_key=True)
        name: Mapped[str] = mapped_column(String(100))
    """
    id: Any         # Базовое поле ID (переопределяется в дочерних классах)
    __name__: str   # Имя класса для автогенерации таблиц

    __allow_unmapped__ = True  # Разрешаем неотмапленные атрибуты (гибкость для развития)

    @declared_attr
    def __tablename__(self) -> str:
        """
        Автоматическая генерация имени таблицы из имени класса.
        
        Правило: ClassName → classname (в нижнем регистре)
        Примеры:
        - MLModel → "mlmodel" 
        - BGTask → "bgtask"
        - UserAccount → "useraccount"
        
        Можно переопределить в дочернем классе:
        class MLModel(Base):
            __tablename__ = "ml_models"  # Кастомное имя вместо "mlmodel"
        """
        return self.__name__.lower()
