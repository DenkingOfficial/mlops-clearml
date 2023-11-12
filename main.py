from scripts.train import train, validation
from scripts.preprocess import download_data, preprocess
from scripts.test import test
from clearml import Task

playground_task: Task = Task.init(
    project_name="tabular-playground-series-aug-2022",
    task_name="tabular-playground-series-aug-2022",
)
playground_task.add_requirements("./requirements.txt")
logger = playground_task.get_logger()

preprocess_parameters = {
    "test_size": 0.2,
    "random_state": 42,
}

train_hyperparameters = {
    "learning_rate": 0.01,
    "max_depth": 3,
    "n_estimators": 100,
    "seed": 42,
}

preprocess_parameters = playground_task.connect(
    preprocess_parameters, name="Preprocess Parameters"
)

train_hyperparameters = playground_task.connect(
    train_hyperparameters, name="Train Hyperparameters"
)

if __name__ == "__main__":
    download_data()
    preprocess(
        preprocess_parameters["test_size"], preprocess_parameters["random_state"]
    )
    model, threshold = train(
        train_hyperparameters["learning_rate"],
        train_hyperparameters["max_depth"],
        train_hyperparameters["n_estimators"],
        train_hyperparameters["seed"],
    )
    roc_auc = validation(model, threshold)
    print(f"ROC AUC: {roc_auc}")
    submission = test()
