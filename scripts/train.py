import pandas as pd
import mlflow
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from constants import DATASET_PATH_PATTERN, MODEL_FILEPATH, RANDOM_STATE, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
from utils import get_logger, load_params

STAGE_NAME = 'train'


mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def train():
    logger = get_logger(logger_name=STAGE_NAME)
    params = load_params(stage_name=STAGE_NAME)

    logger.info('Начали считывать датасеты')
    splits = [None, None, None, None]
    for i, split_name in enumerate(['X_train', 'X_test', 'y_train', 'y_test']):
        splits[i] = pd.read_csv(DATASET_PATH_PATTERN.format(split_name=split_name))
    X_train, X_test, y_train, y_test = splits
    logger.info('Успешно считали датасеты!')

    logger.info('Создаём модель')
    params['random_state'] = RANDOM_STATE

    # mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_params(params)

    logger.info(f'    Параметры модели: {params}')

    # model = LogisticRegression(**params)
    model = RandomForestClassifier(**params)

    logger.info('Обучаем модель')
    model.fit(X_train, y_train)

    logger.info('Сохраняем модель')
    dump(model, MODEL_FILEPATH)
    mlflow.sklearn.log_model(model, artifact_path="model")
    logger.info('Успешно!')


if __name__ == '__main__':
    if not ml_flow.active_run():
        mlflow.start_run(run_name=STAGE_NAME)

    train()

    if mlflow.active_run().info.run_name == STAGE_NAME:
        mlflow.end_run()
