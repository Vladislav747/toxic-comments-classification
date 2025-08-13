"""
Утилиты для обучения и работы с ML/DL моделями.

Содержит вспомогательные классы и функции для:
- Создания и обучения ML pipeline из конфигурации
- Работы с Spacy токенизацией и предобработкой текста
- Загрузки предобученных моделей (ML и DL)
- Сериализации и десериализации моделей
- Выполнения предсказаний на различных типах моделей
"""

from pathlib import Path
import unicodedata

import cloudpickle
import pandas as pd

import spacy
import nltk
from sklearn import pipeline as sk_pipeline

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from transformers import pipeline as tf_pipeline

from serializers import MLModelType, VectorizerType, MLModelConfig
from serializers.utils.trainer import serialize_params
from settings import MODELS_DIR

# === Доступные алгоритмы машинного обучения ===
# Словарь для динамического создания классификаторов по типу из конфигурации
AVAILABLE_ESTIMATORS = {
    MLModelType.logistic_regression: LogisticRegression,  # Логистическая регрессия
    MLModelType.multinomial_nb: MultinomialNB,           # Наивный байесовский классификатор
    MLModelType.linear_svc: LinearSVC                    # Метод опорных векторов (линейный)
}

# === Доступные векторизаторы для обработки текста ===
# Словарь для динамического создания векторизаторов по типу из конфигурации
AVAILABLE_VECTORIZERS = {
    VectorizerType.count_vectorizer: CountVectorizer,    # Подсчет частоты слов (Bag of Words)
    VectorizerType.tfidf_vectorizer: TfidfVectorizer     # TF-IDF векторизация
}


class FunctionWrapper:
    """
    Обертка для функций для поддержки сериализации в multiprocessing.
    
    Проблема: обычные lambda функции не сериализуются через pickle для передачи между процессами.
    Решение: используем cloudpickle для сериализации функции в конструкторе,
    а при вызове десериализуем и выполняем.
    
    Используется для передачи кастомного токенизатора в vectorizer при обучении в отдельном процессе.
    """

    def __init__(self, fn):
        """
        Сериализует функцию для последующей передачи между процессами.
        
        Args:
            fn: Функция для обертки (например, lambda или метод)
        """
        # Сериализуем функцию с помощью cloudpickle (поддерживает lambda и closures)
        self.fn_ser = cloudpickle.dumps(fn)

    def __call__(self, *args, **kwargs):
        """
        Десериализует и выполняет обернутую функцию.
        
        Args:
            *args: Позиционные аргументы для функции
            **kwargs: Именованные аргументы для функции
            
        Returns:
            Результат выполнения обернутой функции
        """
        # Восстанавливаем функцию из сериализованной формы
        fn = cloudpickle.loads(self.fn_ser)
        return fn(*args, **kwargs)


class SpacyTokenizer(BaseEstimator, TransformerMixin):
    """
    Кастомный токенизатор на основе SpaCy для предобработки текста.
    
    Выполняет продвинутую обработку текста:
    - Нормализация Unicode (удаление диакритических знаков)
    - Лемматизация (приведение к базовой форме)
    - Удаление стоп-слов, пунктуации и пробелов
    - Приведение к нижнему регистру
    
    Совместим с sklearn Pipeline (наследует BaseEstimator, TransformerMixin).
    """
    # Загружаем английскую модель SpaCy (требует: python -m spacy download en_core_web_sm)
    nlp = spacy.load('en_core_web_sm')
    # Загружаем список английских стоп-слов из NLTK
    stopwords = set(nltk.corpus.stopwords.words('english'))

    def __init__(self, sep='\t', n_process=1, batch_size=64):
        """
        Инициализация токенизатора с параметрами обработки.
        
        Args:
            sep: Разделитель между токенами в выходной строке
            n_process: Количество процессов для параллельной обработки SpaCy
            batch_size: Размер батча для обработки документов
        """
        self.sep = sep                    # Разделитель токенов (по умолчанию табуляция)
        self.n_process = n_process        # Количество процессов для SpaCy
        self.batch_size = batch_size      # Размер батча для оптимизации памяти

    @staticmethod
    def normalize_text(doc: str) -> str:
        """
        Нормализация текста для удаления диакритических знаков и спецсимволов.
        
        Процесс:
        1. Unicode нормализация NFKD (разложение символов)
        2. Кодирование в ASCII с игнорированием non-ASCII символов
        3. Декодирование обратно в UTF-8
        
        Args:
            doc: Исходный текст для нормализации
            
        Returns:
            Нормализованный текст без диакритических знаков
        
        Example:
            normalize_text('café') -> 'cafe'
            normalize_text('naïve') -> 'naive'
        """
        return unicodedata.normalize(
            'NFKD', doc              # Каноническая декомпозиция
        ).encode(
            'ascii',                 # Конвертируем в ASCII
            'ignore'                 # Игнорируем символы, которые нельзя представить
        ).decode(
            'utf-8',                 # Декодируем обратно в строку
            'ignore'                 # Игнорируем ошибки декодирования
        )

    def fit(self, X, y=None):
        """
        Обучение токенизатора (заглушка для совместимости с sklearn).
        
        SpaCy токенизатор не требует обучения, поэтому просто возвращаем self.
        
        Args:
            X: Входные данные (не используются)
            y: Целевые значения (не используются)
            
        Returns:
            self для метод чейнинга
        """
        return self

    def transform(self, X):
        """
        Преобразует входные тексты в обработанные токенизированные строки.
        
        Процесс обработки каждого документа:
        1. Нормализация Unicode
        2. Токенизация и лемматизация через SpaCy
        3. Фильтрация стоп-слов, пунктуации и пробелов
        4. Приведение к нижнему регистру
        5. Объединение токенов через разделитель
        
        Args:
            X: Список текстов для обработки
            
        Returns:
            Список обработанных строк, где каждая строка содержит
            токены, разделенные символом sep
            
        Example:
            Input: ["Hello, world! How are you?"]
            Output: ["hello\tworld\thow"]
        """
        results = []

        # Создаем пайплайн SpaCy с отключением ненужных компонентов для ускорения
        pipe = self.nlp.pipe(
            map(self.normalize_text, X),        # Предварительная нормализация каждого текста
            disable=['ner', 'parser'],          # Отключаем NER и синтаксический анализ
            batch_size=self.batch_size,         # Обрабатываем батчами для оптимизации памяти
            n_process=self.n_process            # Параллельная обработка
        )

        # Обрабатываем каждый документ
        for doc in pipe:
            # Извлекаем и фильтруем токены
            filtered_tokens = [
                token.lemma_.lower()                # Лемма токена в нижнем регистре
                for token in doc
                if not (                            # Исключаем:
                    token.lemma_.lower() in self.stopwords  # стоп-слова
                    or token.is_space               # пробельные символы
                    or token.is_punct               # пунктуацию
                )
            ]
            # Объединяем токены через разделитель
            results.append(self.sep.join(filtered_tokens))

        return results


def make_pipeline_from_config(
    config: MLModelConfig
) -> tuple[Pipeline, dict, dict]:
    """
    Создает sklearn Pipeline из конфигурации модели.
    
    Функция динамически создает pipeline на основе параметров конфигурации.
    Поддерживает два режима:
    1. Обычная векторизация: vectorizer -> estimator
    2. С SpaCy токенизатором: tokenizer -> vectorizer -> estimator
    
    Args:
        config: Конфигурация модели с параметрами алгоритма и векторизатора
        
    Returns:
        Кортеж из (pipeline, параметры_модели, параметры_векторизатора)
    """
    # Создаем классификатор по типу из конфигурации и применяем параметры
    estimator = AVAILABLE_ESTIMATORS[config.ml_model_type]().set_params(
        **config.ml_model_params
    )
    # Создаем векторизатор по типу из конфигурации и применяем параметры
    vectorizer = AVAILABLE_VECTORIZERS[config.vectorizer_type]().set_params(
        **config.vectorizer_params
    )

    # Проверяем, нужен ли кастомный SpaCy токенизатор
    if config.spacy_lemma_tokenizer:
        # Настраиваем векторизатор для работы с предобработанными токенами
        vectorizer = vectorizer.set_params(
            tokenizer=FunctionWrapper(lambda x: x.split('\t')),  # Токены разделены табами
            strip_accents=None,        # Отключаем автоматическое удаление акцентов
            lowercase=False,           # Отключаем автоматическое приведение к нижнему регистру
            preprocessor=None,         # Отключаем стандартный препроцессор
            stop_words=None,           # Отключаем стандартные стоп-слова
            token_pattern=None         # Отключаем стандартную регулярку токенизации
        )
        # Создаем pipeline с 3 этапами: токенизация -> векторизация -> классификация
        pipe = Pipeline(steps=[
            ('tok', SpacyTokenizer()),     # SpaCy токенизатор с лемматизацией
            ('vec', vectorizer),           # Векторизатор (CountVectorizer или TfidfVectorizer)
            ('estimator', estimator)       # Классификатор (LR, NB, SVC)
        ])
    else:
        # Создаем обычный pipeline с 2 этапами: векторизация -> классификация
        pipe = Pipeline(steps=[
            ('vec', vectorizer),           # Векторизатор со стандартными настройками
            ('estimator', estimator)       # Классификатор
        ])

    # Возвращаем pipeline и параметры для сохранения в БД
    return pipe, estimator.get_params(), vectorizer.get_params()


def train_and_save_model_task(
    model_config: MLModelConfig,
    fit_dataset: pd.DataFrame
) -> tuple[Path, dict, dict]:
    """
    Обучает модель и сохраняет ее на диск.
    
    Функция выполняется в отдельном процессе для изоляции CPU-интенсивных вычислений.
    Обученный pipeline сериализуется через cloudpickle для последующей загрузки.
    
    Args:
        model_config: Конфигурация модели (тип алгоритма, параметры, имя)
        fit_dataset: DataFrame с данными для обучения (колонки: comment_text, toxic)
        
    Returns:
        Кортеж из (путь_к_файлу, параметры_модели, параметры_векторизатора)
        
    Note:
        Функция должна быть сериализуемой для multiprocessing.
    """
    model_name = model_config.name

    # Извлекаем признаки (X) и целевую переменную (y) из датасета
    X = fit_dataset["comment_text"]  # Тексты комментариев
    y = fit_dataset["toxic"]         # Метки токсичности (0 или 1)

    # Создаем pipeline на основе конфигурации
    pipe, model_params, vectorizer_params = make_pipeline_from_config(
        model_config
    )
    
    # Обучаем модель на предоставленных данных
    pipe.fit(X, y)

    # Определяем путь для сохранения модели
    model_file_path = MODELS_DIR / f"{model_name}.cloudpickle"
    
    # Сериализуем обученный pipeline в бинарный файл
    with model_file_path.open('wb') as file:
        cloudpickle.dump(pipe, file)

    # Возвращаем путь к файлу и параметры для сохранения в БД
    return model_file_path, model_params, vectorizer_params


def get_dl_model_predictions(
    model,
    texts: list[str],
    return_scores: bool = False
) -> list[int]:
    """
    Получает предсказания от загруженной DL модели (Transformer).
    
    Обрабатывает выход transformers pipeline для задачи бинарной классификации.
    Поддерживает два режима: получение меток или скоров (вероятностей).
    
    Args:
        model: Загруженная transformers pipeline модель
        texts: Список текстов для классификации
        return_scores: Если True, возвращает скоры, иначе метки классов
        
    Returns:
        Список предсказаний: инты (0/1) или float (скоры) в зависимости от return_scores
        
    Note:
        Ожидает, что модель возвращает labels 'LABEL_0' (нетоксично) и 'LABEL_1' (токсично).
    """
    predictions = []
    
    if return_scores:
        # Режим возврата скоров (вероятностей) для анализа производительности
        results = model(texts, top_k=2)  # Получаем скоры для обоих классов
        for result in results:
            # Ищем скор для класса токсичности (LABEL_1)
            for pred in result:
                if pred["label"] == "LABEL_1":
                    predictions.append(pred["score"])  # Вероятность токсичности
                    break
    else:
        # Режим возврата меток классов (0 или 1)
        results = model(texts)  # Получаем только наиболее вероятный класс
        for result in results:
            # Преобразуем LABEL_1 -> 1, LABEL_0 -> 0
            predictions.append(1 if result["label"] == "LABEL_1" else 0)

    return predictions


def get_dl_model(
    saved_model_path: str,
    tokenizer_name: str
) -> tuple[tf_pipeline, dict, dict]:
    """
    Загружает предобученную DL модель (Transformer) для классификации текста.
    
    Использует transformers библиотеку для создания pipeline для задачи text-classification.
    Извлекает конфигурацию модели и токенизатора для сохранения в БД.
    
    Args:
        saved_model_path: Путь к директории с сохраненной моделью
        tokenizer_name: Имя или путь к токенизатору
        
    Returns:
        Кортеж из (pipeline, параметры_модели, параметры_токенизатора)
    """
    # Создаем transformers pipeline для классификации текста
    pipe = tf_pipeline(
        "text-classification",         # Тип задачи: классификация текста
        model=saved_model_path,           # Путь к предобученной модели
        tokenizer=tokenizer_name          # Имя соответствующего токенизатора
    )
    
    # Извлекаем и сериализуем конфигурацию модели
    model_params = serialize_params(pipe.model.config.to_dict())
    # Извлекаем и сериализуем параметры токенизатора
    vectorizer_params = serialize_params(pipe.tokenizer.init_kwargs)
    
    return pipe, model_params, vectorizer_params


def get_ml_model(saved_model_path: str) -> tuple[sk_pipeline, dict, dict]:
    """
    Загружает сохраненную ML модель с диска.
    
    Десериализует sklearn pipeline, сохраненный через cloudpickle.
    Извлекает параметры классификатора и векторизатора для метаданных.
    
    Args:
        saved_model_path: Путь к .cloudpickle файлу с сохраненной моделью
        
    Returns:
        Кортеж из (pipeline, параметры_классификатора, параметры_векторизатора)
        
    Note:
        Ожидает стандартную структуру pipeline с компонентами:
        'classifier' и 'vectorizer' (или 'vec').
    """
    # Загружаем сериализованный pipeline с диска
    with open(saved_model_path, "rb") as f:
        pipe = cloudpickle.load(f)

    # Извлекаем параметры классификатора и сериализуем
    model_params = serialize_params(
        pipe.named_steps["classifier"].get_params()  # Параметры модели (LR, NB, SVC)
    )
    # Извлекаем параметры векторизатора и сериализуем
    vectorizer_params = serialize_params(
        pipe.named_steps["vectorizer"].get_params()  # Параметры векторизатора (CountVectorizer/TfidfVectorizer)
    )

    return pipe, model_params, vectorizer_params
