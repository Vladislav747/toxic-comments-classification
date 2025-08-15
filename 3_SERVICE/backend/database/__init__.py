# ===== ЭКСПОРТ КОМПОНЕНТОВ БАЗЫ ДАННЫХ =====
# Предоставляет удобный импорт основных компонентов для работы с БД

from database.database import get_db_session, Base

# Экспортируем основные компоненты для использования в других модулях:
# - get_db_session: генератор сессий для FastAPI Depends()
# - Base: базовый класс для SQLAlchemy ORM моделей
__all__ = ['get_db_session', 'Base']

# Использование в других файлах:
# from database import Base, get_db_session
