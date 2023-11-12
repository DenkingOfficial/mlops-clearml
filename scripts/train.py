import xgboost as xgb
from sklearn.metrics import roc_auc_score
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import os
from matplotlib import pyplot as plt


def load_data(validation=False):
    with open("./data/preprocessed/X_train.pkl", "rb") as f:
        X_train = pickle.load(f)
    with open("./data/preprocessed/X_test.pkl", "rb") as f:
        X_test = pickle.load(f)
    with open("./data/preprocessed/y_train.pkl", "rb") as f:
        y_train = pickle.load(f)
    with open("./data/preprocessed/y_test.pkl", "rb") as f:
        y_test = pickle.load(f)
    if validation:
        return X_test, y_test
    else:
        return X_train, y_train


def train(learning_rate: float, max_depth: int, n_estimators: int, seed: int):
    X_train, y_train = load_data()
    xgb_clf = xgb.XGBClassifier(
        objective="binary:logistic",
        learning_rate=learning_rate,
        max_depth=max_depth,
        n_estimators=n_estimators,
        seed=seed,
    )
    xgb_clf.fit(X_train, y_train, verbose=True)
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    xgb_clf.fit(X_train_split, y_train_split, verbose=True)
    y_val_proba = xgb_clf.predict_proba(X_val_split)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_val_split, y_val_proba)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    return xgb_clf, optimal_threshold


def validation(model: xgb.XGBClassifier, threshold: float):
    X_test, y_test = load_data(validation=True)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()
    return roc_auc_score(y_test, y_pred)


def main():
    model, threshold = train()
    print(f"ROC AUC: {validation(model, threshold)}")
    os.makedirs("./data/models", exist_ok=True)
    with open("./data/models/model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("./data/models/threshold.pkl", "wb") as f:
        pickle.dump(threshold, f)
