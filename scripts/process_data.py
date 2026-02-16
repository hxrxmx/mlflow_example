import os
import numpy as np
import pandas as pd
import mlflow
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from constants import DATASET_NAME, DATASET_PATH_PATTERN, TEST_SIZE, RANDOM_STATE, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
from utils import get_logger, load_params

STAGE_NAME = 'process_data'


mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def process_data():
    logger = get_logger(logger_name=STAGE_NAME)
    params = load_params(stage_name=STAGE_NAME)

    logger.info('Начали скачивать данные')
    dataset = load_dataset(DATASET_NAME)
    logger.info('Успешно скачали данные!')

    logger.info('Делаем предобработку данных')
    df = dataset['train'].to_pandas()
    columns = params['features']
    target_column = 'income'
    X, y = df[columns], df[target_column]

    mlflow.log_params({
        "features": params['features'],
        "random_state": RANDOM_STATE,
    })

    logger.info(f'    Используемые фичи: {columns}')

    all_cat_features = [
        'workclass', 'education', 'marital.status', 'occupation', 'relationship',
        'race', 'sex', 'native.country',
    ]
    cat_features = list(set(columns) & set(all_cat_features))
    num_features = list(set(columns) - set(all_cat_features))

    preprocessor = OrdinalEncoder()
    X_transformed = np.hstack([X[num_features], preprocessor.fit_transform(X[cat_features])])
    y_transformed: pd.Series = (y == '>50K').astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y_transformed, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    X_train = X_train[:params['train_size']]
    y_train = y_train[:params['train_size']]

    mlflow.log_param("train_size", len(y_train))
    mlflow.log_param("test_size", len(y_test))

    logger.info(f'    Размер тренировочного датасета: {len(y_train)}')
    logger.info(f'    Размер тестового датасета: {len(y_test)}')

    logger.info('Начали сохранять датасеты')
    os.makedirs(os.path.dirname(DATASET_PATH_PATTERN), exist_ok=True)
    for split, split_name in zip(
        (X_train, X_test, y_train, y_test),
        ('X_train', 'X_test', 'y_train', 'y_test'),
    ):
        file_path = DATASET_PATH_PATTERN.format(split_name=split_name)
        pd.DataFrame(split).to_csv(file_path, index=False)
        mlflow.log_artifact(file_path, artifact_path="datasets")
    logger.info('Успешно сохранили датасеты!')


if __name__ == '__main__':
    if not mlflow.active_run():
        mlflow.start_run(run_name=STAGE_NAME)

    process_data()

    if mlflow.active_run().info.run_name == STAGE_NAME:
        mlflow.end_run()
