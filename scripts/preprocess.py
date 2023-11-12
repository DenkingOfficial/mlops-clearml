import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pickle
import os
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile

api = KaggleApi()
api.authenticate()


def download_data():
    data_dir = "./data"

    train_data_path = os.path.join(data_dir, "train.csv")
    test_data_path = os.path.join(data_dir, "test.csv")

    if os.path.exists(train_data_path) and os.path.exists(test_data_path):
        return

    os.makedirs(data_dir, exist_ok=True)

    competition_name = "tabular-playground-series-aug-2022"

    api.competition_download_files(competition_name, data_dir)

    competition_zip = os.path.join(data_dir, f"{competition_name}.zip")
    with zipfile.ZipFile(competition_zip, "r") as zip_ref:
        zip_ref.extractall(data_dir)

    os.remove(competition_zip)


def preprocess(test_size, random_state):
    df_train = pd.read_csv("./data/train.csv", index_col="id")
    df_test = pd.read_csv("./data/test.csv", index_col="id")

    target = df_train["failure"]
    df_train = df_train.drop("failure", axis=1)

    df_total = pd.concat([df_train, df_test], axis=0)
    numerical_columns = df_total.select_dtypes(exclude=["object"]).columns
    imputer = SimpleImputer(strategy="median")

    scaler = RobustScaler()

    pipeline = Pipeline(steps=[("imputer", imputer), ("scaler", scaler)])

    pipeline.fit(df_total[numerical_columns])

    df_total[numerical_columns] = pipeline.transform(df_total[numerical_columns])
    df_total.select_dtypes(include="object").columns
    df_total = pd.get_dummies(df_total)
    size_df1 = len(df_train)
    size_df2 = len(df_test)

    df_train_prep = df_total.iloc[:size_df1]
    df_test_prep = df_total.iloc[size_df1 : size_df1 + size_df2]

    X_train, X_test, y_train, y_test = train_test_split(
        df_train_prep, target, test_size=test_size, random_state=random_state
    )

    os.makedirs("./data/preprocessed", exist_ok=True)
    with open("./data/preprocessed/X_train.pkl", "wb") as f:
        pickle.dump(X_train, f)
    with open("./data/preprocessed/X_test.pkl", "wb") as f:
        pickle.dump(X_test, f)
    with open("./data/preprocessed/y_train.pkl", "wb") as f:
        pickle.dump(y_train, f)
    with open("./data/preprocessed/y_test.pkl", "wb") as f:
        pickle.dump(y_test, f)

    df_test_prep.to_parquet("./data/preprocessed/test.parquet")


if __name__ == "__main__":
    download_data()
    preprocess()
