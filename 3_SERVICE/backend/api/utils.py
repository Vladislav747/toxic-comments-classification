# ===== УТИЛИТЫ ДЛЯ РАБОТЫ С ЗАГРУЖАЕМЫМИ ФАЙЛАМИ =====
# Этот модуль содержит вспомогательные функции для обработки файлов,
# загружаемых пользователями через API endpoints

import zipfile

import pandas as pd
from fastapi import HTTPException, UploadFile
from starlette import status


def extract_dataset_from_zip_file(uploaded_file: UploadFile) -> pd.DataFrame:
    """
    Извлекает датасет из ZIP-архива для обучения/предсказания моделей.
    
    Функция выполняет следующие проверки и операции:
    1. Проверяет что файл имеет расширение .zip
    2. Извлекает список CSV файлов из архива
    3. Проверяет что в архиве ровно один CSV файл
    4. Читает CSV в pandas DataFrame
    
    Args:
        uploaded_file (UploadFile): Загруженный пользователем ZIP-файл
        
    Returns:
        pd.DataFrame: Датасет для машинного обучения
        
    Raises:
        HTTPException 400: Если файл не .zip, нет CSV, или несколько CSV
        
    Использование:
        # В endpoint обучения модели:
        dataset = extract_dataset_from_zip_file(fit_file)
        # → получаем DataFrame с колонками comment_text и toxic
    """
    
    # Проверка расширения файла - должен быть ZIP-архив
    if not uploaded_file.filename.endswith(".zip"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Файл должен быть zip-архивом."
        )

    # Открываем ZIP-архив и ищем CSV файлы внутри
    with zipfile.ZipFile(uploaded_file.file) as zf:
        # Получаем список всех .csv файлов в архиве
        csv_files = [name for name in zf.namelist() if name.endswith(".csv")]

        # Валидация: должен быть ровно один CSV файл
        if len(csv_files) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="CSV-файл не найден в архиве."
            )
        if len(csv_files) > 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Архив должен содержать только один CSV-файл."
            )

        # Читаем единственный CSV файл в pandas DataFrame
        with zf.open(csv_files[0]) as csv_file:
            dataset = pd.read_csv(csv_file)
            # Ожидаемая структура датасета:
            # - comment_text: текст комментария (строка)
            # - toxic: метка токсичности (0 или 1)

    return dataset
