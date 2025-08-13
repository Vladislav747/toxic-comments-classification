"""
Настройки логирования и путей проекта.

Конфигурирует систему логирования с ротацией файлов и выводом в консоль.
Определяет ключевые пути проекта.
"""

import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

# === Определение ключевых путей проекта ===
BASE_DIR = Path(__file__).parent.parent.resolve()          # Корневая директория backend
MODELS_DIR = BASE_DIR / "models"                            # Директория для хранения ML моделей
LOG_FILE_PATH = BASE_DIR / 'logs' / 'backend' / 'backend.log'  # Путь к файлу логов

# === Настройка форматирования логов ===
# Формат для файловых логов: время + уровень + сообщение
file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
# Формат для консоли: только уровень + сообщение 
stream_formatter = logging.Formatter("%(levelname)s:     %(message)s")

# === Настройка обработчика файловых логов ===
# Автоматическая ротация логов каждую полночь с сохранением 7 файлов
timed_handler = TimedRotatingFileHandler(
    LOG_FILE_PATH,      # Путь к файлу логов
    when="midnight",    # Ротация каждую полночь
    interval=1,         # Каждые 1 день
    backupCount=7,      # Хранить 7 бэкапов (неделя)
    delay=True          # Отложенное создание файла
)
timed_handler.setLevel(logging.INFO)           # Минимальный уровень INFO
timed_handler.setFormatter(file_formatter)     # Применяем формат с временем

# === Настройка обработчика консольных логов ===
# Вывод логов в стандартный поток вывода (stdout)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)          # Минимальный уровень INFO
stream_handler.setFormatter(stream_formatter)   # Применяем упрощенный формат

# === Создание и конфигурация основного логгера ===
# Основной логгер приложения с двойным выводом: файл + консоль
logger = logging.getLogger("toxic_comments_app")
logger.setLevel(logging.INFO)                  # Общий уровень логирования
logger.addHandler(timed_handler)               # Добавляем запись в файл
logger.addHandler(stream_handler)              # Добавляем вывод в консоль
