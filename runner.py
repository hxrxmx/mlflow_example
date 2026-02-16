import mlflow
from scripts import evaluate, process_data, train
from constants import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME


if __name__ == '__main__':
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="randomforest"):
        process_data()
        train()
        evaluate()
