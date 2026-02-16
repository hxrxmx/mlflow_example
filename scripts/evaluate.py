import os

import numpy as np
import pandas as pd
import mlflow
import matplotlib.pyplot as plt
from joblib import load
from sklearn.metrics import get_scorer, confusion_matrix, ConfusionMatrixDisplay

from constants import DATASET_PATH_PATTERN, MODEL_FILEPATH, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
from utils import get_logger, load_params

STAGE_NAME = 'evaluate'


mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def evaluate():
    logger = get_logger(logger_name=STAGE_NAME)
    params = load_params(stage_name=STAGE_NAME)

    logger.info('Начали считывать датасеты')
    splits = [None, None, None, None]
    for i, split_name in enumerate(['X_train', 'X_test', 'y_train', 'y_test']):
        splits[i] = pd.read_csv(DATASET_PATH_PATTERN.format(split_name=split_name))
    X_train, X_test, y_train, y_test = splits
    logger.info('Успешно считали датасеты!')
    
    logger.info('Загружаем обученную модель')
    if not os.path.exists(MODEL_FILEPATH):
        raise FileNotFoundError(
            'Не нашли файл с моделью. Убедитесь, что был запущен шаг с обучением'
        )
    model = load(MODEL_FILEPATH)

    logger.info('Считываем предсказания')
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = np.where(y_proba >= 0.5, 1, 0)

    logger.info('Скорим модель на тесте')
    metrics = {}
    for metric_name in params['metrics']:
        scorer = get_scorer(metric_name)
        score = scorer(model, X_test, y_test)
        metrics[metric_name] = score
    mlflow.log_metrics(metrics)
    logger.info(f'Значения метрик - {metrics}')

    logger.info("Сохраняем пикчи")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    
    plot_path = "confusion_matrix.png"
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path)
    os.remove(plot_path)


if __name__ == '__main__':
    if not mlflow.active_run():
        mlflow.start_run(run_name=STAGE_NAME)
    
    evaluate()
    
    if mlflow.active_run().info.run_name == STAGE_NAME:
        mlflow.end_run()
