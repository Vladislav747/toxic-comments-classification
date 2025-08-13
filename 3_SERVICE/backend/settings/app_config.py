"""
Конфигурация приложения с использованием Pydantic Settings.

Обеспечивает типизированную конфигурацию с валидацией и загрузкой из .env файла.
"""

import multiprocessing

from pydantic import field_validator, conint
from pydantic_settings import BaseSettings, SettingsConfigDict

# Глобальный счетчик активных процессов для ограничения параллельности
# Используется multiprocessing.Value для thread-safe операций
active_processes = multiprocessing.Value('i', 1)


class AppConfig(BaseSettings):
    """
    Основная конфигурация приложения.
    
    Наследуется от BaseSettings для автоматической загрузки из переменных окружения.
    Поддерживает валидацию типов и значений с помощью Pydantic.
    """

    # === Настройки производительности ===
    cores_cnt: conint(gt=1) = 2              # Максимальное количество CPU ядер для обучения
    models_max_cnt: int = 2                  # Максимальное количество загруженных моделей
    max_saved_bg_tasks: conint(gt=2) = 10    # Максимальное количество сохраненных фоновых задач

    # === Настройки базы данных PostgreSQL ===
    postgres_host: str = 'localhost'         # Хост сервера БД
    postgres_port: int = 5432                # Порт сервера БД  
    postgres_user: str = 'postgres'          # Пользователь БД
    postgres_db: str = 'toxic_comments'      # Название базы данных
    postgres_password: str = 'postgres'      # Пароль пользователя БД
    postgres_driver: str = 'postgresql+asyncpg'  # Драйвер для асинхронного подключения

    # Конфигурация Pydantic: загрузка из .env файла, игнорирование лишних полей
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    @property
    def db_url(self) -> str:
        """
        Формирует строку подключения к базе данных.
        
        Returns:
            Строка подключения в формате: postgresql+asyncpg://user:pass@host:port/db
        """
        return (
            f'{self.postgres_driver}://{self.postgres_user}:'
            f'{self.postgres_password}@{self.postgres_host}:'
            f'{self.postgres_port}/{self.postgres_db}'
        )

    @field_validator('cores_cnt', mode='before')
    def set_cores_cnt(cls, v) -> int:
        """
        Валидирует и ограничивает количество используемых CPU ядер.
        
        Гарантирует, что количество ядер не превышает доступное на системе.
        Предотвращает перегрузку системы при обучении моделей.
        
        Args:
            v: Запрашиваемое количество ядер
            
        Returns:
            Валидное количество ядер (не больше доступных в системе)
        """
        available_cores = multiprocessing.cpu_count()
        return min(int(v), available_cores)


# Глобальный экземпляр конфигурации приложения
# Автоматически загружается из переменных окружения и .env файла
app_config = AppConfig()
