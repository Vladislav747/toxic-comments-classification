# ===== УТИЛИТЫ СЕРИАЛИЗАЦИИ ДЛЯ ПАРАМЕТРОВ МОДЕЛЕЙ =====
# Вспомогательные функции для преобразования объектов Python в JSON-совместимый формат

from typing import Any

import numpy as np


def serialize_params(obj: Any) -> Any:
    """
    Рекурсивно сериализует объект в JSON-совместимый формат.
    
    Основная проблема: параметры ML моделей часто содержат numpy типы,
    callable объекты и другие специальные типы Python, которые не могут
    быть напрямую сериализованы в JSON для сохранения в базе данных.
    
    Функция обрабатывает:
    - Примитивные типы (int, float, str, bool, None) → возвращает как есть
    - Коллекции (list, tuple, dict) → рекурсивно обрабатывает элементы
    - numpy типы (integers, floats, arrays) → преобразует в стандартные Python типы
    - Типы и функции → возвращает их имена как строки
    - Все остальное → преобразует в строку
    
    Примеры использования:
    
    # Numpy типы:
    serialize_params(np.int64(42)) → 42
    serialize_params(np.array([1, 2, 3])) → [1, 2, 3]
    
    # Параметры sklearn моделей:
    serialize_params({'C': np.float64(1.0), 'solver': 'liblinear'}) 
    → {'C': 1.0, 'solver': 'liblinear'}
    
    # Callable объекты:
    serialize_params({'tokenizer': str.lower}) → {'tokenizer': 'lower'}
    
    Используется при сохранении конфигурации моделей в базу данных.
    """
    # Примитивные типы - возвращаем как есть
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    
    # Списки и кортежи - рекурсивно обрабатываем каждый элемент
    if isinstance(obj, (list, tuple)):
        return [serialize_params(item) for item in obj]
    
    # Словари - рекурсивно обрабатываем значения
    if isinstance(obj, dict):
        return {key: serialize_params(value) for key, value in obj.items()}
    
    # numpy целые числа → стандартный int
    if isinstance(obj, np.integer):
        return int(obj)
    
    # numpy числа с плавающей точкой → стандартный float
    if isinstance(obj, np.floating):
        return float(obj)
    
    # numpy массивы → списки Python
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    
    # Типы (классы) → имя типа как строка
    if isinstance(obj, type):
        return obj.__name__
    
    # Callable объекты (функции, методы) → имя как строка
    if callable(obj):
        return obj.__name__
    
    # Все остальное → строковое представление
    return str(obj)
