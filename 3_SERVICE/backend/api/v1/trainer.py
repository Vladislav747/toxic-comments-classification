# ===== ОСНОВНОЙ API РОУТЕР ДЛЯ УПРАВЛЕНИЯ МОДЕЛЯМИ =====
# Этот модуль содержит все REST API endpoints для:
# - Получения списка моделей: GET /api/v1/models/
# - Обучения новых моделей: POST /api/v1/models/fit
# - Загрузки/выгрузки моделей в память: POST /api/v1/models/load, /unload
# - Предсказаний: POST /api/v1/models/predict/{name}
# - Получения метрик: POST /api/v1/models/predict_scores/
# - Удаления моделей: DELETE /api/v1/models/remove/{name}, /remove_all

import io
import json
from ast import Param
from typing import Annotated

from fastapi import (
    APIRouter,
    HTTPException,
    Depends,
    UploadFile,
    File,
    Form,
    Path
)
from http import HTTPStatus

from pydantic import ValidationError
from starlette import status
from starlette.responses import StreamingResponse

from api.utils import extract_dataset_from_zip_file
from dependency import get_trainer_service
from exceptions import (
    ModelNameAlreadyExistsError,
    ModelNotFoundError,
    ModelNotLoadedError,
    ModelsLimitExceededError,
    InvalidFitPredictDataError,
    ActiveProcessesLimitExceededError,
    DefaultModelRemoveUnloadError,
    ModelNotTrainedError, ModelAlreadyLoadedError
)
from serializers import (
    MLModelConfig,
    LoadRequest,
    MessageResponse,
    UnloadRequest,
    PredictResponse,
    MLModelType,
    MLModelInListResponse,
    PredictRequest,
    VectorizerType
)
from services import TrainerService

router = APIRouter()


# ===== ENDPOINT: ПОЛУЧЕНИЕ СПИСКА МОДЕЛЕЙ =====
@router.get(
    "/",
    response_model=list[MLModelInListResponse],
    description="Получение списка моделей"
)
async def get_models(
    trainer_service: Annotated[TrainerService, Depends(get_trainer_service)],  # ЗАВИСИМОСТЬ
    is_dl: Annotated[bool | None, Param()] = None  # ВАЛИДАЦИЯ параметра query - Фильтр: True=DL модели, False=ML, None=все
):
    """
    Возвращает список всех доступных моделей в системе.
    
    Модели могут быть:
    - Предустановленные (default_*) - всегда доступны
    - Пользовательские - созданные через /fit
    
    Фильтрация:
    - is_dl=True: только глубокие модели (BERT, DistilBERT, etc.)
    - is_dl=False: только классические ML модели (LogReg, SVM, etc.)
    - is_dl=None: все модели
    
    Возвращает информацию о каждой модели:
    - Имя и тип модели
    - Статус (trained/training/failed)
    - Загружена ли в память для инференса
    """
    return await trainer_service.get_models(is_dl=is_dl)


# ===== ENDPOINT: ОБУЧЕНИЕ НОВОЙ МОДЕЛИ =====
@router.post(
    "/fit",
    status_code=HTTPStatus.OK,
    response_model=MessageResponse,
    description="Обучение новой модели"
)
async def fit(
    # Annotated = современный способ аннотации типов в FastAPI 
    # Формат: Annotated[Тип, Зависимость/Валидация]
    # Заменяет старый синтаксис: param: Type = Depends(...)
    trainer_service: Annotated[TrainerService, Depends(get_trainer_service)],  
    # Depends() = система внедрения зависимостей FastAPI
    # get_trainer_service() создает экземпляр TrainerService с подключением к БД
    # FastAPI автоматически вызывает функцию и передает результат в параметр
    fit_file: Annotated[
        UploadFile,
        File(description=(
            "ZIP-архив с CSV-файлом. Файл должен содержать 2 столбца: "
            "`comment_text` (сырой текст) "
            "и `toxic` (бинарная метка токсичности)"
        ))
    ],  # ZIP-архив с обучающими данными:
        # - Внутри: один CSV файл с заголовками
        # - Столбцы: comment_text (текст), toxic (0/1)
        # - Формат: "Привет!" → 0, "Дурак!" → 1
        # - Рекомендуется: 1000+ строк, сбалансированные классы
    name: Annotated[str, Form()],  # ВАЛИДАЦИЯ form-data - Уникальное имя модели
    vectorizer_type: Annotated[VectorizerType, Form()],  # TF-IDF или BOW
    # Form() = валидация для multipart/form-data (не зависимость!)
    # Указывает что параметр приходит из HTML формы, а не JSON
    ml_model_type: Annotated[MLModelType, Form()],  # Form() - ВАЛИДАЦИЯ form-data - Тип ML модели: LR, SVM, NB, etc.
    ml_model_params: Annotated[
        str,
        Form(description="Валидная JSON-строка")
    ] = "{}",  # ВАЛИДАЦИЯ form-data + ДЕФОЛТ "{}" - Параметры модели (C, max_iter, и т.д.)
    spacy_lemma_tokenizer: Annotated[bool, Form()] = False,  # ВАЛИДАЦИЯ form-data + ДЕФОЛТ False - Использовать ли spaCy для лемматизации
    vectorizer_params: Annotated[
        str,
        Form(description="Валидная JSON-строка")
    ] = "{}"  # ВАЛИДАЦИЯ form-data + ДЕФОЛТ "{}" - Параметры векторайзера (max_features, ngram_range, и т.д.)
):
    """
    Запускает обучение новой модели классификации токсичных комментариев.
    
    Процесс обучения:
    1. Извлекаем датасет из ZIP-архива
    2. Валидируем JSON-параметры
    3. Создаем конфигурацию модели
    4. Запускаем обучение в фоновом процессе
    """
    # Парсим JSON-параметры для модели и векторайзера
    try:
        parsed_ml_model_params = json.loads(ml_model_params)
        parsed_vectorizer_params = json.loads(vectorizer_params)
    except json.decoder.JSONDecodeError:
        # Ошибка парсинга JSON - некорректный формат
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Поля 'ml_model_params' и 'vectorizer_params' должны быть "
                "валидными JSON-строками."
            )
        )

    # Извлекаем датасет из загруженного ZIP-файла
    dataset = extract_dataset_from_zip_file(fit_file)

    # Запускаем обучение модели с обработкой ошибок
    try:
        return await trainer_service.fit_models(
            MLModelConfig(
                name=name,
                vectorizer_type=vectorizer_type,
                spacy_lemma_tokenizer=spacy_lemma_tokenizer,
                vectorizer_params=parsed_vectorizer_params,
                ml_model_type=ml_model_type,
                ml_model_params=parsed_ml_model_params
            ),
            dataset
        )
    # Python проверяет except'ы СВЕРХУ ВНИЗ
    # Находит ПЕРВЫЙ подходящий по типу исключения
    # Выполняет ТОЛЬКО его код
    # Остальные except'ы игнорируются
    # Продолжает выполнение ПОСЛЕ всего try/except блока
    except ValidationError as e:
        # ПЕРВЫЙ except: Ошибка валидации Pydantic-модели
        # ValidationError имеет сложную структуру - нужна специальная обработка
        # e.errors() возвращает список ошибок, берем первую и извлекаем сообщение
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=repr(e.errors()[0]["msg"])  # Специальное извлечение сообщения из Pydantic
        )
    except (
            ModelNameAlreadyExistsError,      # Имя модели уже существует
            InvalidFitPredictDataError,       # Некорректные данные для обучения
            ActiveProcessesLimitExceededError # Превышен лимит процессов обучения
    ) as e:
        # ВТОРОЙ except: Наши кастомные исключения (из exceptions.py)
        # У них уже есть готовое поле .detail с понятным сообщением на русском
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.detail  # Простое использование готового сообщения
        )


# ===== ENDPOINT: ЗАГРУЗКА МОДЕЛИ В ПАМЯТЬ =====
@router.post(
    "/load",
    response_model=list[MessageResponse],
    description="Загрузка модели в пространство инференса"
)
async def load(
    request: LoadRequest,  # ВАЛИДАЦИЯ JSON body - Содержит имя модели для загрузки
    trainer_service: Annotated[TrainerService, Depends(get_trainer_service)]  # ЗАВИСИМОСТЬ
):
    """
    Загружает обученную модель в оперативную память для быстрых предсказаний.
    
    Процесс загрузки:
    1. Проверяем что модель существует
    2. Проверяем что она обучена (есть файл)
    3. Проверяем лимит моделей в памяти
    4. Загружаем модель в loaded_models
    """
    try:
        return await trainer_service.load_model(request.name)
    except ModelNotFoundError as e:
        # Модель не найдена в системе
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.detail
        )
    except (
        ModelNotTrainedError,     # Модель еще не обучена
        ModelsLimitExceededError, # Превышен лимит моделей в памяти
        ModelAlreadyLoadedError   # Модель уже загружена
    ) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.detail
        )


# ===== ENDPOINT: ВЫГРУЗКА МОДЕЛИ ИЗ ПАМЯТИ =====
@router.post(
    "/unload",
    response_model=list[MessageResponse],
    description="Выгрузка модели из пространства инференса"
)
async def unload(
    request: UnloadRequest,  # ВАЛИДАЦИЯ JSON body - Содержит имя модели для выгрузки
    trainer_service: Annotated[TrainerService, Depends(get_trainer_service)]  # ЗАВИСИМОСТЬ
):
    """
    Выгружает модель из оперативной памяти, освобождая ресурсы.
    
    Процесс выгрузки:
    1. Проверяем что модель существует
    2. Проверяем что она загружена в память
    3. Проверяем что это не default-модель
    4. Удаляем из loaded_models
    """
    try:
        return await trainer_service.unload_model(request.name)
    except ModelNotFoundError as e:
        # Модель не найдена в системе
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.detail
        )
    except (
        ModelNotLoadedError,            # Модель не загружена в память
        DefaultModelRemoveUnloadError   # Попытка выгрузить предустановленную модель
    ) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.detail
        )


# ===== ENDPOINT: ПРЕДСКАЗАНИЕ МОДЕЛИ =====
@router.post(
    "/predict/{name}",
    response_model=PredictResponse,
    description="Предсказание модели"
)
async def predict(
    name: Annotated[str, Path()],  # ВАЛИДАЦИЯ URL path - Имя модели в URL пути
    request: PredictRequest,       # ВАЛИДАЦИЯ JSON body - Текст(ы) для классификации
    trainer_service: Annotated[TrainerService, Depends(get_trainer_service)]  # ЗАВИСИМОСТЬ
):
    """
    Выполняет предсказание токсичности для переданного текста(ов).
    
    Процесс предсказания:
    1. Проверяем что модель существует
    2. Проверяем что она загружена в память
    3. Преобразуем текст(ы) через векторайзер
    4. Получаем предсказания от модели
    """
    try:
        return await trainer_service.predict(name, request)
    except ModelNotFoundError as e:
        # Модель не найдена в системе
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.detail
        )
    except (
        ModelNotLoadedError,        # Модель не загружена в память
        InvalidFitPredictDataError  # Некорректные данные для предсказания
    ) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.detail
        )


# ===== ENDPOINT: ПОЛУЧЕНИЕ МЕТРИК МОДЕЛЕЙ =====
@router.post(
    "/predict_scores/",
    response_class=StreamingResponse,
    description="Получение данных для построения кривых обучения",
    response_description="CSV-файл с данными для построения кривых обучения"
)
async def predict_scores(
    names: Annotated[
        str,
        Form(description="Список имен моделей через запятую (model_1,model_2)")
    ],  # ВАЛИДАЦИЯ form-data - Перечисление моделей для сравнения
    predict_file: Annotated[
        UploadFile,
        File(description=(
            "ZIP-архив с CSV-файлом. Файл должен содержать 2 столбца: "
            "`comment_text` (сырой текст) и "
            "`toxic` (бинарная метка токсичности)"
        ))
    ],  # ВАЛИДАЦИЯ file upload - Тестовый датасет с разметкой
    trainer_service: Annotated[TrainerService, Depends(get_trainer_service)]  # ЗАВИСИМОСТЬ
):
    """
    Выполняет предсказания несколькими моделями и возвращает CSV с метриками.
    
    Процесс:
    1. Извлекаем тестовый датасет из ZIP
    2. Парсим список имен моделей
    3. Для каждой модели выполняем предсказания
    4. Вычисляем метрики качества (accuracy, precision, recall, F1)
    5. Формируем CSV с результатами
    
    Используется для:
    - Сравнения качества разных моделей
    - Построения ROC-кривых
    - Анализа производительности
    """
    # Извлекаем тестовый датасет с разметкой
    dataset = extract_dataset_from_zip_file(predict_file)
    try:
        # Выполняем предсказания и вычисляем метрики
        result = await trainer_service.predict_scores(
            names.split(","),  # Разбиваем строку на список имен
            dataset
        )

        # ===== ИНТЕРЕСНЫЙ ПАТТЕРН: CSV в памяти без файла на диске =====
        # 1. Создаем буфер в памяти (как виртуальный файл)
        buffer = io.StringIO()
        # 2. Записываем DataFrame прямо в буфер как CSV
        result.to_csv(buffer, index=False)  # index=False убирает номера строк
        # 3. Перематываем указатель в начало буфера (как rewind кассеты)
        buffer.seek(0)  # Важно! Иначе StreamingResponse прочитает пустоту
        
        # 4. Возвращаем буфер как HTTP стрим - браузер скачает как файл
        return StreamingResponse(
            buffer,                     # Источник данных (наш буфер с CSV)
            media_type="text/csv",      # MIME-тип говорит браузеру что это CSV
            headers={
                "Content-Disposition": (
                    "attachment; filename=predicted_scores.csv"  # Имя файла при скачивании
                )
            }
        )
        # Результат: пользователь получает CSV файл, а на сервере никаких файлов не создается!
    except ModelNotFoundError as e:
        # Модель не найдена в системе
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.detail
        )
    except (
        ModelNotLoadedError,        # Модель не загружена в память
        InvalidFitPredictDataError  # Некорректные данные
    ) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.detail
        )


# ===== ENDPOINT: УДАЛЕНИЕ ОДНОЙ МОДЕЛИ =====
@router.delete(
    "/remove/{name}",
    response_model=list[MessageResponse],
    description="Удаление модели"
)
async def remove(
    name: Annotated[str, Path()],  # ВАЛИДАЦИЯ URL path - Имя модели для удаления
    trainer_service: Annotated[TrainerService, Depends(get_trainer_service)]  # ЗАВИСИМОСТЬ
):
    """
    Полностью удаляет модель из системы.
    
    Процесс удаления:
    1. Проверяем что модель существует
    2. Проверяем что это не предустановленная модель
    3. Выгружаем из памяти (если загружена)
    4. Удаляем файлы модели с диска
    5. Удаляем запись из базы данных
    """
    try:
        return await trainer_service.remove_model(name)
    except ModelNotFoundError as e:
        # Модель не найдена в системе
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.detail
        )
    except DefaultModelRemoveUnloadError as e:
        # Попытка удалить предустановленную модель
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.detail
        )


# ===== ENDPOINT: УДАЛЕНИЕ ВСЕХ МОДЕЛЕЙ =====
@router.delete(
    "/remove_all",
    response_model=MessageResponse,
    description="Удаление всех моделей"
)
async def remove_all(
    trainer_service: Annotated[TrainerService, Depends(get_trainer_service)]
):
    """
    Удаляет ВСЕ пользовательские модели из системы.
    
    Процесс:
    1. Находим все пользовательские модели (не default)
    2. Для каждой модели выполняем:
       - Выгрузка из памяти
       - Удаление файлов
       - Удаление из БД
    3. Очищаем папку моделей
    
    ОСОБОЕ ВНИМАНИЕ: это опасная операция!
    Удаляет все обученные пользователем модели.
    Предустановленные модели остаются нетронутыми.
    """
    return await trainer_service.remove_all_models()
