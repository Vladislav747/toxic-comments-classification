# ===== КАСТОМНЫЕ ИСКЛЮЧЕНИЯ ДЛЯ УПРАВЛЕНИЯ МОДЕЛЯМИ =====

class ModelNotFoundError(Exception):
    """
    Исключение: модель не найдена в системе.
    
    Возникает когда:
    - Пытаемся загрузить несуществующую модель
    - Обращаемся к модели которая еще не создана
    - Модель была удалена из системы
    """
    def __init__(self, model_id: str):
        self.detail = (
            f"Модель '{model_id}' не найдена. Возможно она еще обучается."
        )
        # Дружелюбное сообщение пользователю на русском языке
        # Подсказывает возможную причину - модель может быть в процессе обучения
        super().__init__(self.detail)
        # Вызываем конструктор родительского класса Exception

    # Пример использования:
    # raise ModelNotFoundError("my_custom_model")
    # → "Модель 'my_custom_model' не найдена. Возможно она еще обучается."


class ModelNameAlreadyExistsError(Exception):
    """
    Исключение: попытка создать модель с уже существующим именем.
    
    Возникает когда:
    - Пытаемся создать модель с именем которое уже занято
    - Защищает от перезаписи существующих моделей
    """
    def __init__(self, model_id: str):
        self.detail = f"Модель '{model_id}' уже существует."
        super().__init__(self.detail)
    
    # Пример:
    # raise ModelNameAlreadyExistsError("my_model")
    # → "Модель 'my_model' уже существует."


class ModelAlreadyLoadedError(Exception):
    """
    Исключение: попытка загрузить уже загруженную модель.
    
    Возникает когда:
    - Модель уже находится в памяти (loaded_models)
    - Пытаемся повторно вызвать /load для той же модели
    - Защищает от дублирования моделей в памяти
    """
    def __init__(self, model_id: str):
        self.detail = f"Модель '{model_id}' уже загружена."
        super().__init__(self.detail)
    
    # Пример:
    # if model_name in loaded_models:
    #     raise ModelAlreadyLoadedError(model_name)


class ModelNotTrainedError(Exception):
    """
    Исключение: попытка использовать необученную модель.
    
    Возникает когда:
    - Пытаемся загрузить модель которая еще не завершила обучение
    - Обучение модели завершилось с ошибкой
    - Файл модели не был создан
    """
    def __init__(self, model_id: str):
        self.detail = f"Модель '{model_id}' еще не обучилась."
        super().__init__(self.detail)
    
    # Типичный сценарий:
    # 1. POST /api/v1/models/fit - запуск обучения
    # 2. POST /api/v1/models/load - попытка загрузить до завершения обучения
    # 3. → ModelNotTrainedError


class ModelNotLoadedError(Exception):
    """
    Исключение: попытка использовать модель которая не загружена в память.
    
    Возникает когда:
    - Пытаемся делать предсказания с моделью не в loaded_models
    - Модель была выгружена из памяти (unload)
    - Забыли загрузить модель перед использованием
    """
    def __init__(self, model_id: str):
        self.detail = f"Модель '{model_id}' не загружена в память."
        super().__init__(self.detail)
    
    # Правильная последовательность:
    # 1. POST /api/v1/models/load - загружаем в память
    # 2. POST /api/v1/models/predict - используем для предсказаний
    # Если пропустили шаг 1 → ModelNotLoadedError


# ===== ИСКЛЮЧЕНИЯ ОГРАНИЧЕНИЙ РЕСУРСОВ =====

class ModelsLimitExceededError(Exception):
    """
    Исключение: превышен лимит моделей в памяти.
    
    Возникает когда:
    - В loaded_models уже максимальное количество моделей
    - Пытаемся загрузить еще одну модель
    - Защищает от переполнения RAM
    """
    detail = "Превышен лимит моделей для инференса."
    # Статическое сообщение - не зависит от параметров
    # Лимит задается в app_config.models_max_cnt (по умолчанию 2)

    # Решение: выгрузить ненужные модели через /unload


class ActiveProcessesLimitExceededError(Exception):
    """
    Исключение: превышен лимит активных процессов обучения.
    
    Возникает когда:
    - Уже запущено максимальное количество процессов обучения
    - Пытаемся запустить еще одно обучение
    - Защищает от перегрузки CPU
    """
    detail = "Превышен лимит активных процессов."
    # Лимит процессов = app_config.cores_cnt - 1
    # Если cores_cnt=2, то максимум 1 процесс обучения одновременно

    # Решение: дождаться завершения текущих обучений


# ===== ИСКЛЮЧЕНИЯ ЗАЩИТЫ СИСТЕМЫ =====

class DefaultModelRemoveUnloadError(Exception):
    """
    Исключение: попытка удалить или выгрузить предустановленную модель.
    
    Возникает когда:
    - Пытаемся удалить модель из DEFAULT_MODELS_INFO
    - Пытаемся выгрузить default модель из памяти
    - Защищает критически важные модели системы
    """
    detail = "Нельзя удалить или выгрузить из памяти модели по умолчанию."
    
    # Защищенные модели:
    # - default_logistic_regression
    # - default_linear_svc  
    # - default_multinomial_naive_bayes
    # - default_distilbert
    # - default_deberta_v3


# ===== ИСКЛЮЧЕНИЯ ВАЛИДАЦИИ ДАННЫХ =====

class InvalidFitPredictDataError(Exception):
    """
    Исключение: некорректные данные для обучения или предсказания.
    
    Возникает когда:
    - Неправильный формат входных данных
    - Отсутствуют обязательные столбцы
    - Пустые или поврежденные данные
    """
    def __init__(self, message: str):
        self.detail = message  # Пользовательское сообщение об ошибке
        super().__init__(self.detail)
    
    # Примеры:
    # raise InvalidFitPredictDataError("Отсутствует столбец 'comment_text'")
    # raise InvalidFitPredictDataError("Пустой датасет")
    # raise InvalidFitPredictDataError("Неподдерживаемый формат файла")