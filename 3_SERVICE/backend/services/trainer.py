"""
Основной сервис для обучения и управления ML/DL моделями.

Центральный компонент системы, отвечающий за:
- Асинхронное обучение моделей в фоновых процессах
- Загрузку/выгрузку моделей в память
- Выполнение предсказаний
- Управление жизненным циклом моделей
- Мониторинг производительности и ресурсов
"""

import asyncio
import datetime as dt
import os

import cloudpickle
import pandas as pd
from fastapi import BackgroundTasks

from exceptions import (
    ModelNameAlreadyExistsError,
    ModelNotFoundError,
    ModelNotLoadedError,
    ModelsLimitExceededError,
    DefaultModelRemoveUnloadError,
    ModelNotTrainedError,
    ActiveProcessesLimitExceededError,
    ModelAlreadyLoadedError
)
from repository import BgTasksRepository, ModelsRepository
from serializers import (
    BGTaskSchema,
    BGTaskStatus,
    MessageResponse,
    PredictResponse,
    MLModelConfig,
    PredictRequest,
    MLModelInListResponse,
    MLModelCreateSchema,
)
from serializers.utils.trainer import serialize_params
from services import BGTasksService
from services.utils.trainer import (
    train_and_save_model_task,
    get_dl_model_predictions,
    get_dl_model,
    get_ml_model,
)
from settings import active_processes, app_config, logger, MODELS_DIR
from store import DEFAULT_MODELS_INFO


class TrainerService:
    """
    Главный сервис для обучения и управления ML/DL моделями.
    
    Этот сервис является центральным компонентом системы машинного обучения.
    Координирует все операции с моделями: от обучения до развертывания.
    
    Основные возможности:
    - Асинхронное обучение моделей с контролем ресурсов
    - Управление загруженными в память моделями
    - Выполнение предсказаний на загруженных моделях
    - Автоматическая инициализация моделей по умолчанию
    - Мониторинг фоновых задач обучения
    - Безопасное удаление пользовательских моделей
    """

    def __init__(
        self,
        models_repo: ModelsRepository,
        loaded_models: dict,
        bg_tasks_repo: BgTasksRepository,
        background_tasks: BackgroundTasks,
        bg_tasks_service: BGTasksService
    ):
        """
        Инициализация сервиса с внедрением зависимостей.
        
        Args:
            models_repo: Репозиторий для работы с моделями в БД
            loaded_models: Глобальный словарь загруженных в память моделей (UUID -> model)
            bg_tasks_repo: Репозиторий для работы с фоновыми задачами
            background_tasks: FastAPI механизм для фоновых задач
            bg_tasks_service: Сервис для управления фоновыми задачами
        """
        # Получаем пул процессов из глобального состояния приложения
        # Используется для CPU-интенсивных задач обучения моделей
        # при инициализации класса TrainerService мы получаем  пул процессов (Используется для CPU-интенсивных задач обучения моделей)  из глобального состояния приложения 
        from main import app
        self.process_executor = app.state.process_executor

        # Внедренные зависимости для работы с данными и координации
        self.models_repo = models_repo              # Доступ к моделям в БД
        self.loaded_models = loaded_models          # Кеш загруженных моделей в памяти
        self.bg_tasks_repo = bg_tasks_repo          # Управление фоновыми задачами
        self.background_tasks = background_tasks    # FastAPI фоновые задачи
        self.bg_tasks_service = bg_tasks_service    # Бизнес-логика фоновых задач

    async def fit_models(
        self,
        model_config: MLModelConfig,
        fit_dataset: pd.DataFrame
    ) -> MessageResponse:
        """
        Запускает процесс обучения новой ML модели в фоновом режиме.
        
        Основная точка входа для обучения моделей. Выполняет предварительные проверки,
        создает фоновую задачу и запускает асинхронное обучение в отдельном процессе.
        Контролирует ограничения по количеству одновременно выполняющихся процессов.
        
        Процесс обучения:
        1. Проверка ограничений по ресурсам (количество активных процессов)
        2. Проверка уникальности имени модели
        3. Запуск фоновой задачи обучения
        4. Немедленный возврат ответа пользователю (асинхронность)
        
        Args:
            model_config: Конфигурация модели (тип алгоритма, параметры, имя)
            fit_dataset: Данные для обучения с колонками 'comment_text' и 'toxic'
            
        Returns:
            Сообщение о том, что обучение запущено
            
        Raises:
            ActiveProcessesLimitExceededError: Превышен лимит активных процессов
            ModelNameAlreadyExistsError: Модель с таким именем уже существует
        """
        # === Проверка ограничений системы ===
        # Контролируем количество одновременно обучающихся моделей для предотвращения перегрузки CPU
        if active_processes.value >= app_config.cores_cnt:
            raise ActiveProcessesLimitExceededError()

        model_name = model_config.name

        # === Проверка уникальности имени модели ===
        # Предотвращаем конфликты имен в базе данных
        if await self.models_repo.get_model_by_name(model_name):
            raise ModelNameAlreadyExistsError(model_name)

        # === Запуск фонового обучения ===
        # Добавляем задачу в FastAPI background tasks для асинхронного выполнения - просто добавляем НЕ ЗАПУСКАЕМ
        # Это позволяет немедленно вернуть ответ пользователю, не ожидая завершения обучения
        self.background_tasks.add_task(
            self._execute_fitting_task,     # Метод для выполнения обучения
            model_name, model_config, fit_dataset  # Параметры для обучения
        )

        # Возвращаем подтверждение о запуске процесса обучения
        return MessageResponse(message=(
            f"Запущено обучение модели '{model_config.name}'."
        ))

    async def _execute_fitting_task(
        self,
        model_name: str,
        model_config: MLModelConfig,
        fit_dataset: pd.DataFrame
    ) -> None:
        """
        Выполняет процесс обучения модели в фоновом режиме.
        
        Этот метод запускается как фоновая задача и координирует весь жизненный цикл обучения:
        - Создание записи о задаче в БД для мониторинга
        - Запуск CPU-интенсивного обучения в отдельном процессе
        - Обработка результатов и ошибок
        - Обновление статуса задачи в БД
        - Управление счетчиком активных процессов
        
        Процесс выполняется с таймаутом (30 минут) для предотвращения зависших задач.
        В случае ошибки автоматически очищает созданные записи в БД.
        
        Args:
            model_name: Уникальное имя модели для идентификации
            model_config: Конфигурация алгоритма и параметров
            fit_dataset: Обучающие данные
            
        Note:
            Метод обновляет глобальный счетчик active_processes для контроля нагрузки.
            При любом исходе (успех/ошибка) счетчик уменьшается на 1.
        """
        # Получаем текущий event loop для управления асинхронными операциями
        loop = asyncio.get_event_loop()

        # === Создание записей для мониторинга ===
        # Создаем запись о фоновой задаче для отслеживания прогресса пользователем
        bg_task_id = await self.bg_tasks_repo.create_task(
            BGTaskSchema(name=f"Обучение модели '{model_name}'")
        )
        # Создаем запись о модели в БД (пока не обученной)
        await self.models_repo.create_model(MLModelCreateSchema(
            name=model_name,
            type=model_config.ml_model_type
        ))

        # === Увеличиваем счетчик активных процессов ===
        # Используется для контроля нагрузки на систему
        # # задавали в settings.py active_processes = multiprocessing.Value('i', 1)
        active_processes.value += 1
        try:
            # ========================================================================
            # ПОДРОБНОЕ ОБЪЯСНЕНИЕ asyncio.wait_for():
            # ========================================================================
            # 
            # asyncio.wait_for() - это функция-обертка, которая:
            # 1. Запускает awaitable операцию (в нашем случае - run_in_executor)
            # 2. Параллельно запускает внутренний таймер на указанное время
            # 3. Ждет, что произойдет первым:
            #    a) Операция завершится успешно → возвращает результат
            #    b) Таймер истечет → выбрасывает asyncio.TimeoutError
            # 
            # ЗАЧЕМ ЭТО НУЖНО:
            # - Без таймаута: если процесс обучения "зависнет", задача будет ждать вечно
            # - С таймаутом: через 30 минут мы гарантированно получим контроль обратно
            # - Это позволяет освободить ресурсы и уведомить пользователя об ошибке
            # 
            # ЧТО ПРОИСХОДИТ ПРИ ТАЙМАУТЕ:
            # 1. asyncio.wait_for() выбрасывает asyncio.TimeoutError  
            # 2. Выполнение переходит в блок except
            # 3. Процесс обучения в отдельном процессе может продолжать работать,
            #    но мы больше не ждем его завершения
            # 4. active_processes.value уменьшается, освобождая слот для новых задач
            # 
            # АЛЬТЕРНАТИВЫ БЕЗ wait_for():
            # await loop.run_in_executor(...)  # ❌ Может зависнуть навсегда
            # 
            # С wait_for():
            # await asyncio.wait_for(           # ✅ Гарантированно завершится за 30 минут
            #     loop.run_in_executor(...), 
            #     timeout=1800
            # )
            # ========================================================================
            # === Запуск обучения в отдельном процессе ===
            # Используем ProcessPoolExecutor для изоляции CPU-интенсивных вычислений
            # от основного asyncio event loop. Это предотвращает блокировку API.
            # === Получение результата выполненной операции обучения ===
            # Эти переменные содержат РЕЗУЛЬТАТ выполнения функции:
            # train_and_save_model_task(model_config, fit_dataset) -> tuple[Path, dict, dict]
            # которая была запущена в отдельном процессе и успешно завершилась
            # 
            # ВАЖНО: Мы получаем эти значения только если:
            # 1. Процесс обучения завершился успешно (без исключений)
            # 2. Время выполнения не превысило 30 минут (timeout=1800)
            # 3. Функция train_and_save_model_task вернула корректный результат
            #
            # Если произойдет любая ошибка - выполнение перейдет в блок except
            (
                model_file_path,     # Путь к сохраненному файлу модели (возвращено из train_and_save_model_task)
                model_params,        # Параметры обученной модели для БД (возвращено из train_and_save_model_task)
                vectorizer_params    # Параметры векторизатора для БД (возвращено из train_and_save_model_task)
            ) = await asyncio.wait_for(
                # === Внутренняя асинхронная операция ===
                # run_in_executor возвращает awaitable объект, который завершится
                # когда отдельный процесс закончит выполнение train_and_save_model_task
                loop.run_in_executor(
                    self.process_executor,      # Пул процессов из состояния приложения
                    train_and_save_model_task,  # Функция обучения (выполняется в отдельном процессе)
                    model_config, fit_dataset   # Аргументы для функции обучения
                ),
                # === Критический таймаут для предотвращения зависших процессов ===
                timeout=1800  # 30 минут = максимальное время обучения модели
                # Почему именно 30 минут:
                # - Достаточно для обучения больших моделей на средних датасетах
                # - Предотвращает бесконечное ожидание при зависших процессах
                # - Освобождает ресурсы системы от "мертвых" задач
                # - Позволяет пользователю получить обратную связь об ошибке
            )

            # === Обновление модели после успешного обучения ===
            # Сохраняем информацию об обученной модели в БД
            await self.models_repo.update_model_after_training(
                model_name=model_name,
                is_trained=True,                                    # Помечаем как обученную
                # === СЕРИАЛИЗАЦИЯ ПАРАМЕТРОВ ДЛЯ СОХРАНЕНИЯ В БД ===
                # ПРОБЛЕМА: model_params и vectorizer_params содержат сложные Python объекты:
                # - numpy arrays, функции, lambda, классы sklearn
                # - Например: {'tokenizer': <function at 0x12345>, 'max_features': 1000}
                # PostgreSQL НЕ МОЖЕТ напрямую сохранить такие объекты в JSON поле
                #
                # РЕШЕНИЕ: serialize_params() преобразует сложные объекты в простые типы:
                # - Функции → строки с именем функции
                # - numpy arrays → обычные списки  
                # - Объекты классов → словари с их атрибутами
                # - Результат: только str, int, float, bool, list, dict
                #
                # ПРИМЕР ПРЕОБРАЗОВАНИЯ:
                # ДО:  {'tokenizer': <function>, 'max_features': 1000, 'vocab': array([...])}
                # ПОСЛЕ: {'tokenizer': 'custom_tokenizer', 'max_features': 1000, 'vocab': [...]}
                model_params=serialize_params(model_params),        # Python dict → JSON-совместимый dict
                vectorizer_params=serialize_params(vectorizer_params),  # Python dict → JSON-совместимый dict
                saved_model_file_path=str(model_file_path),         # Path объект → строка для БД
            )

            # === Подготовка успешного результата ===
            status = BGTaskStatus.success
            result_msg = (
                f"Модель '{model_name}' успешно обучена."
            )
            # Уменьшаем счетчик активных процессов
            active_processes.value -= 1
        except Exception as e:
            # === Обработка ошибок обучения ===
            # Удаляем запись о модели из БД, так как обучение не завершилось
            await self.models_repo.delete_model(model_name)

            status = BGTaskStatus.failure
            
            # === Специальная обработка таймаута (превышение 30 минут) ===
            # TimeoutError может быть как asyncio.TimeoutError, так и обычным TimeoutError
            # В данном контексте это всегда будет asyncio.TimeoutError от wait_for()
            if isinstance(e, TimeoutError):
                result_msg = (
                    f"Превышено время обучения модели ({model_name}). "
                    "Задача остановлена."
                )
                # ВАЖНО: таймаут логируем как INFO, а не ERROR, потому что:
                # 1. Это не ошибка приложения, а защитный механизм
                # 2. Возможно, модель просто требует больше времени
                # 3. Пользователь может попробовать снова с упрощенными параметрами
                logger.info(result_msg)
                
                # ТЕХНИЧЕСКАЯ ДЕТАЛЬ: 
                # Процесс обучения в отдельном процессе может продолжать работать,
                # даже после таймаута. ProcessPoolExecutor не может принудительно
                # остановить уже запущенный процесс. Но это не критично, так как:
                # - Ресурсы в конечном итоге освободятся при завершении процесса
                # - Счетчик active_processes уже уменьшен, новые задачи могут запускаться
                # - Файл модели (если создастся) будет проигнорирован из-за удаления записи из БД
            else:
                # Любая другая ошибка (недостаток памяти, некорректные данные, etc.)
                result_msg = f"Ошибка при обучении модели '{model_name}': {e}."
                logger.error(result_msg)  # Логируем как ошибку

            # Обязательно уменьшаем счетчик активных процессов
            active_processes.value -= 1

        # === Обновление статуса фоновой задачи ===
        # Обновляем информацию о задаче для отображения пользователю
        await self.bg_tasks_repo.update_task(
            task_uuid=bg_task_id,           # ID задачи для обновления
            status=status,                  # Финальный статус (success/failure)
            result_msg=result_msg,          # Сообщение с результатом или ошибкой
            updated_at=dt.datetime.now()    # Время завершения
        )
        
        # === Очистка старых задач ===
        # Автоматически удаляем старые завершенные задачи для оптимизации БД
        await self.bg_tasks_service.rotate_tasks()

    async def load_model(self, model_name: str) -> list[MessageResponse]:
        """
        Загружает обученную модель в оперативную память для выполнения предсказаний.
        
        Процесс загрузки:
        1. Проверка существования модели в БД
        2. Проверка статуса обученности модели  
        3. Проверка лимитов загруженных моделей
        4. Десериализация модели с диска
        5. Сохранение в глобальном кеше loaded_models
        6. Обновление статуса is_loaded в БД
        
        Args:
            model_name: Имя модели для загрузки из БД
            
        Returns:
            Список с сообщением о успешной загрузке
            
        Raises:
            ModelNotFoundError: Модель не найдена в БД
            ModelNotTrainedError: Модель не обучена
            ModelAlreadyLoadedError: Модель уже загружена в память
            ModelsLimitExceededError: Превышен лимит загруженных моделей
        """
        # === Валидация модели и ограничений ===
        model = await self.models_repo.get_model_by_name(model_name)
        
        # Проверяем, что модель существует в БД
        if not model:
            raise ModelNotFoundError(model_name)
            
        # Проверяем, что модель не загружена уже в память
        if model.uuid in self.loaded_models:
            raise ModelAlreadyLoadedError(model_name)
            
        # Проверяем лимит одновременно загруженных моделей
        # (пользовательские модели + модели по умолчанию)
        if len(self.loaded_models) >= (
            app_config.models_max_cnt + len(DEFAULT_MODELS_INFO)
        ):
            raise ModelsLimitExceededError()
            
        # Проверяем, что модель обучена и готова к использованию
        if not model.is_trained:
            raise ModelNotTrainedError(model_name)

        # === Загрузка модели с диска в память ===
        # Десериализуем сохраненный sklearn/transformers pipeline
        with open(model.saved_model_file_path, 'rb') as f:
            self.loaded_models[model.uuid] = cloudpickle.load(f)

        # === Обновление статуса в БД ===
        # Помечаем модель как загруженную для корректного отображения в UI
        await self.models_repo.update_model_is_loaded(model_name, True)

        return [MessageResponse(
            message=f"Модель '{model_name}' загружена в память."
        )]

    async def unload_model(self, model_name: str) -> list[MessageResponse]:
        """
        Выгружает модель из оперативной памяти, сохраняя запись в БД.
        
        Освобождает память, занятую загруженной моделью, но оставляет возможность
        загрузить её снова позже. Модель остается в базе данных и на диске.
        
        Используется для управления памятью при работе с большим количеством моделей
        или при необходимости освободить ресурсы.
        
        Args:
            model_name: Имя модели для выгрузки из памяти
            
        Returns:
            Список с сообщением об успешной выгрузке
            
        Raises:
            ModelNotFoundError: Модель не найдена в БД
            ModelNotLoadedError: Модель не загружена в память
            DefaultModelRemoveUnloadError: Попытка выгрузить системную модель
            
        Note:
            Системные модели (DEFAULT_MODELS_INFO) защищены от выгрузки.
            После выгрузки модель можно снова загрузить методом load_model().
        """
        # === Валидация возможности выгрузки ===
        model = await self.models_repo.get_model_by_name(model_name)
        
        # Проверяем существование модели в БД
        if not model:
            raise ModelNotFoundError(model_name)
            
        # Проверяем, что модель загружена в память
        if model.uuid not in self.loaded_models:
            raise ModelNotLoadedError(model_name)
            
        # Защита системных моделей от выгрузки
        if model_name in DEFAULT_MODELS_INFO:
            raise DefaultModelRemoveUnloadError()

        # === Выгрузка модели из памяти ===
        # Удаляем из кеша загруженных моделей (освобождаем RAM)
        self.loaded_models.pop(model.uuid, None)
        
        # Обновляем статус в БД (помечаем как выгруженную)
        await self.models_repo.update_model_is_loaded(model_name, False)
        
        return [MessageResponse(
            message=f"Модель '{model_name}' выгружена из памяти."
        )]

    async def predict(
        self,
        model_name: str,
        predict_data: PredictRequest
    ) -> PredictResponse:
        """
        Выполняет предсказания токсичности текстов с использованием загруженной модели.
        
        Основной метод для инференса модели. Поддерживает как традиционные ML модели
        (sklearn), так и современные DL модели (transformers). Автоматически определяет
        тип модели и применяет соответствующий алгоритм предсказания.
        
        Процесс предсказания:
        1. Валидация существования и загруженности модели
        2. Получение модели из кеша loaded_models  
        3. Определение типа модели (ML/DL)
        4. Выполнение предсказания соответствующим методом
        5. Возврат результатов в стандартизированном формате
        
        Args:
            model_name: Имя модели для выполнения предсказаний
            predict_data: Данные для предсказания (список текстов в поле X)
            
        Returns:
            Объект с предсказаниями в виде списка (0 - не токсично, 1 - токсично)
            
        Raises:
            ModelNotFoundError: Модель не найдена в БД
            ModelNotLoadedError: Модель не загружена в память
            
        Note:
            Для DL моделей используется специальная функция get_dl_model_predictions
            для корректной обработки выхода transformers pipeline.
        """
        # === Валидация модели ===
        model = await self.models_repo.get_model_by_name(model_name)

        # Проверяем существование модели в БД
        if not model:
            raise ModelNotFoundError(model_name)
            
        # Проверяем, что модель загружена в память (доступна для предсказаний)
        if model.uuid not in self.loaded_models:
            raise ModelNotLoadedError(model_name)

        # === Получение модели из кеша ===
        # Извлекаем загруженную модель по UUID из глобального словаря
        loaded_model = self.loaded_models.get(model.uuid)
        
        # === Выполнение предсказания в зависимости от типа модели ===
        if model.is_dl_model:
            # Для DL моделей (transformers): используем специальную функцию
            # которая корректно обрабатывает выход pipeline и возвращает метки классов
            predictions = get_dl_model_predictions(
                loaded_model,           # Transformers pipeline
                predict_data.X          # Список текстов для классификации
            )
        else:
            # Для традиционных ML моделей (sklearn): прямой вызов метода predict
            # sklearn pipeline автоматически применит векторизацию и классификацию
            predictions = loaded_model.predict(predict_data.X).tolist()

        # Возвращаем результаты в стандартизированном формате
        return PredictResponse(predictions=predictions)

    async def get_models(
        self,
        is_dl: bool | None = None
    ) -> list[MLModelInListResponse]:
        """
        Получает список всех моделей или моделей определенного типа с их параметрами.
        
        Возвращает полную информацию о моделях в системе, включая их статус
        (обученная/загруженная) и сериализованные параметры для отображения в UI.
        
        Args:
            is_dl: Фильтр по типу модели:
                - True: только DL модели (transformers)  
                - False: только ML модели (sklearn)
                - None: все модели (по умолчанию)
                
        Returns:
            Список объектов MLModelInListResponse с информацией о каждой модели:
            - uuid, name, type: идентификация модели
            - is_trained, is_loaded: статусы модели
            - model_params, vectorizer_params: параметры в JSON-формате
            
        Note:
            Параметры автоматически сериализуются для безопасной передачи в UI.
            Используется для отображения списка моделей в веб-интерфейсе.
        """
        # === Формирование списка моделей для ответа ===
        response_list = []

        # Получаем модели из БД с опциональной фильтрацией по типу
        for model_info in await self.models_repo.get_models(is_dl=is_dl):
            # === Создание объекта ответа для каждой модели ===
            response_list.append(
                MLModelInListResponse(
                    uuid=model_info.uuid,
                    name=model_info.name,
                    type=model_info.type,
                    is_trained=model_info.is_trained,      # Статус обученности
                    is_loaded=model_info.is_loaded,        # Статус загруженности в память
                    # === Сериализация параметров для UI ===
                    # Преобразуем сложные Python объекты в JSON-совместимый формат
                    model_params=serialize_params(model_info.model_params),
                    vectorizer_params=serialize_params(
                        model_info.vectorizer_params
                    )
                )
            )

        return response_list

    async def predict_scores(
        self,
        model_names: list[str],
        predict_dataset: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Получает оценки (scores/вероятности) предсказаний от нескольких моделей для анализа производительности.
        
        Этот метод используется для сравнения производительности различных моделей на одном датасете.
        В отличие от обычного predict(), возвращает не метки классов (0/1), а непрерывные оценки
        вероятности токсичности, что позволяет построить ROC-кривые и вычислить метрики качества.
        
        Процесс выполнения:
        1. Валидация существования и загруженности всех запрошенных моделей
        2. Для каждой модели выполнение предсказаний с получением scores
        3. Объединение результатов в единый DataFrame для анализа
        
        Args:
            model_names: Список имен моделей для сравнения
            predict_dataset: DataFrame с колонками 'comment_text' (тексты) и 'toxic' (true labels)
            
        Returns:
            DataFrame с колонками:
            - 'model_name': имя модели
            - 'scores': вероятности/оценки токсичности (float от 0 до 1)
            - 'y_true': истинные метки (0/1)
            
        Raises:
            ModelNotFoundError: Одна из моделей не найдена в БД
            ModelNotLoadedError: Одна из моделей не загружена в память
            
        Note:
            Используется для построения ROC-кривых, вычисления AUC и других метрик качества.
            Для DL моделей получает вероятности через return_scores=True.
            Для ML моделей использует predict_proba() или decision_function().
        """
        # === Получение информации о моделях из БД ===
        models = await self.models_repo.get_models_by_names(model_names)

        # === Валидация существования всех запрошенных моделей ===
        # Проверяем, что все модели из списка найдены в БД
        for model_name in model_names:
            if model_name not in [db_model.name for db_model in models]:
                raise ModelNotFoundError(model_name)
                
        # === Валидация загруженности всех моделей ===
        # Проверяем, что все модели загружены в память и доступны для предсказаний
        for db_model in models:
            if db_model.uuid not in self.loaded_models:
                raise ModelNotLoadedError(db_model.name)

        # === Выполнение предсказаний для каждой модели ===
        results = []
        for db_model in models:
            # Получаем загруженную модель из кеша
            # возвращает sklearn Pipeline объект
            loaded_model = self.loaded_models.get(db_model.uuid)

            # Извлекаем данные из датасета
            X = predict_dataset["comment_text"]  # Тексты для предсказания
            y_true = predict_dataset["toxic"]    # Истинные метки для сравнения

            # === Получение scores в зависимости от типа модели ===
            if db_model.is_dl_model:
                # Для DL моделей (transformers): используем специальную функцию с флагом return_scores
                # Возвращает вероятности токсичности (float от 0 до 1)
                scores = get_dl_model_predictions(
                    loaded_model,           # Transformers pipeline
                    X.tolist(),            # Конвертируем pandas Series в список
                    return_scores=True     # Важно! Возвращаем scores, а не метки классов
                )
            else:
                # Для традиционных ML моделей (sklearn): проверяем доступные методы
                if hasattr(loaded_model, "predict_proba"):
                    # predict_proba() возвращает вероятности для каждого класса: [[p0, p1], ...]
                    # Берем второй столбец ([:, 1]) - вероятность класса "токсично"
                    scores = loaded_model.predict_proba(X)[:, 1]
                else:
                    # decision_function() возвращает "расстояние" до разделяющей гиперплоскости
                    # Положительные значения = токсично, отрицательные = не токсично
                    # Не нормализовано к [0,1], но подходит для ROC-анализа
                    scores = loaded_model.decision_function(X)

            # === Создание DataFrame с результатами для текущей модели ===
            results.append(pd.DataFrame({
                "model_name": db_model.name,  # Имя модели для идентификации
                "scores": scores,             # Оценки/вероятности токсичности
                "y_true": y_true             # Истинные метки для вычисления метрик
            }))

        # === Объединение результатов всех моделей в один DataFrame ===
        # Это позволяет легко сравнивать модели и строить графики
        return pd.concat(results, ignore_index=True)

    async def remove_model(self, model_name: str) -> list[MessageResponse]:
        """
        Полное удаление пользовательской модели из системы.
        
        Выполняет комплексное удаление модели:
        1. Удаление записи из базы данных
        2. Удаление из кеша загруженных моделей (оперативная память)
        3. Удаление файла модели с диска
        
        Защищает системные модели по умолчанию от случайного удаления.
        
        Args:
            model_name: Имя модели для удаления
            
        Returns:
            Список с сообщением об успешном удалении
            
        Raises:
            ModelNotFoundError: Модель не найдена в БД
            DefaultModelRemoveUnloadError: Попытка удалить системную модель
            
        Note:
            Модели по умолчанию (из DEFAULT_MODELS_INFO) защищены от удаления.
            Операция необратима - восстановить модель можно только переобучив заново.
        """
        # === Валидация возможности удаления ===
        model = await self.models_repo.get_model_by_name(model_name)

        # Проверяем существование модели в БД
        if not model:
            raise ModelNotFoundError(model_name)
            
        # Защита системных моделей от удаления
        if model_name in DEFAULT_MODELS_INFO:
            raise DefaultModelRemoveUnloadError()

        # === Комплексное удаление модели ===
        # Сохраняем путь к файлу для последующего удаления
        saved_model_filepath = model.saved_model_file_path
        
        # 1. Удаляем запись из базы данных
        await self.models_repo.delete_model(model_name)
        
        # .pop метод удаления элемента из словаря с двумя аргументами:
        # 2. Удаляем из кеша загруженных моделей (освобождаем RAM)  
        # ВАЖНО: используем model.uuid, а не model_name как ключ в loaded_models
        self.loaded_models.pop(model.uuid, None)  # None = не выбрасывать исключение если ключа нет
        
        # 3. Удаляем файл модели с диска (освобождаем дисковое пространство)
        if os.path.isfile(saved_model_filepath):
            os.remove(saved_model_filepath)

        return [MessageResponse(message=f"Модель '{model_name}' удалена.")]

    async def remove_all_models(self) -> MessageResponse:
        """
        Массовое удаление всех пользовательских моделей из системы.
        
        Выполняет полную очистку системы от пользовательских моделей, сохраняя
        только системные модели по умолчанию. Операция включает:
        1. Удаление всех записей из БД (кроме DEFAULT_MODELS_INFO)
        2. Очистка кеша загруженных моделей 
        3. Удаление всех файлов моделей с диска
        
        Используется для "сброса к заводским настройкам" или освобождения места.
        
        Returns:
            Сообщение о количестве удаленных моделей
            
        Note:
            Системные модели (DEFAULT_MODELS_INFO) остаются нетронутыми.
            Операция необратима - все пользовательские модели будут потеряны.
            Рекомендуется создать резервную копию важных моделей перед выполнением.
        """
        # === Массовое удаление пользовательских моделей ===
        # Репозиторий автоматически исключает модели из DEFAULT_MODELS_INFO
        deleted_models = await self.models_repo.delete_all_user_models()
        
        # === Очистка файлов и кеша для каждой удаленной модели ===
        for model in deleted_models:
            # Удаляем файл модели с диска (если существует)
            file_path = model.saved_model_file_path
            if file_path and os.path.isfile(file_path):
                os.remove(file_path)
                
            # Удаляем из кеша загруженных моделей (освобождаем RAM)
            self.loaded_models.pop(model.uuid, None)

        return MessageResponse(
            message="Все модели, кроме моделей по умолчанию, удалены."
        )

    async def create_and_load_models(self) -> None:
        """
        Инициализация и загрузка системных моделей по умолчанию при запуске приложения.
        
        Этот метод выполняется при старте сервера и обеспечивает готовность базовых моделей
        к работе. Обрабатывает как предобученные ML модели (sklearn), так и DL модели (transformers).
        
        Процесс инициализации:
        1. Итерация по всем моделям из DEFAULT_MODELS_INFO
        2. Загрузка каждой модели с диска (ML/DL)
        3. Создание/обновление записей в БД
        4. Загрузка в кеш для немедленного использования
        5. Восстановление ранее загруженных пользовательских моделей
        
        Note:
            Модели по умолчанию должны быть предварительно обучены и сохранены в директории models/default/.
            Метод автоматически определяет тип модели (ML/DL) и применяет соответствующий загрузчик.
            Пользовательские модели с флагом is_loaded=True также восстанавливаются в память.
        """
        # === Инициализация системных моделей по умолчанию ===
        for model_name, model_info in DEFAULT_MODELS_INFO.items():
            # Формируем путь к файлу модели в директории по умолчанию
            saved_model_path = MODELS_DIR / "default" / model_info["filename"]
            is_dl_model = model_info["is_dl_model"]

            # === Загрузка модели в зависимости от типа ===
            if is_dl_model:
                # Для DL моделей (transformers): загружаем с указанием токенизатора
                pipe, model_params, vectorizer_params = get_dl_model(
                    saved_model_path,           # Путь к директории с моделью
                    model_info["tokenizer"]     # Имя соответствующего токенизатора
                )
            else:
                # Для традиционных ML моделей (sklearn): загружаем из cloudpickle файла
                pipe, model_params, vectorizer_params = get_ml_model(
                    saved_model_path           # Путь к .cloudpickle файлу
                )

            # === Создание/обновление записи в БД ===
            db_model = await self.models_repo.get_model_by_name(model_name)
            if db_model:
                # Модель уже существует в БД - используем существующий UUID
                db_model_uuid = db_model.uuid
            else:
                # Создаем новую запись в БД для модели по умолчанию
                db_model_uuid = await self.models_repo.create_model(
                    MLModelCreateSchema(
                        name=model_name,
                        type=model_info["type"],
                        is_dl_model=is_dl_model,
                        is_trained=True,                    # Модели по умолчанию уже обучены
                        is_loaded=True,                     # Помечаем как загруженную
                        model_params=model_params,
                        vectorizer_params=vectorizer_params,
                        saved_model_file_path=saved_model_path
                    )
                )

            # === Загрузка в кеш для немедленного использования ===
            self.loaded_models[db_model_uuid] = pipe

        # === Восстановление ранее загруженных пользовательских моделей ===
        # После перезапуска сервера восстанавливаем в память все модели,
        # которые были загружены до остановки (is_loaded=True)
        for db_model in await self.models_repo.get_models():
            # Проверяем, что это пользовательская модель (не системная) и она была загружена
            if db_model.name not in DEFAULT_MODELS_INFO and db_model.is_loaded:
                # Десериализуем модель с диска и загружаем в кеш
                with open(db_model.saved_model_file_path, "rb") as f:
                    # десериализуем модель с диска и загружаем в кеш
                    self.loaded_models[db_model.uuid] = cloudpickle.load(f)
