import pandas as pd
import pickle
from datetime import datetime
from kaggle.api.kaggle_api_extended import KaggleApi
import time

api = KaggleApi()
api.authenticate()


def test():
    df_test = pd.read_parquet("./data/preprocessed/test.parquet")
    model = pickle.load(open("./data/models/model.pkl", "rb"))
    threshold = pickle.load(open("./data/models/threshold.pkl", "rb"))
    submission = pd.read_csv("./data/sample_submission.csv")
    test_proba = model.predict_proba(df_test)[:, 1]
    test_pred = (test_proba >= threshold).astype(int)
    test_pred = pd.DataFrame(test_pred)
    submission["failure"] = test_pred
    now = datetime.now()
    datetime_str = now.strftime("%d-%m-%Y-%H-%M-%S")
    submission.to_csv(f"./data/submission-{datetime_str}.csv", index=False)
    api.competition_submit(
        f"./data/submission-{datetime_str}.csv",
        f"My submission on {datetime_str}",
        "tabular-playground-series-aug-2022",
    )
    submissions: list[dict] = api.competitions_submissions_list(
        "tabular-playground-series-aug-2022", async_req=True
    ).get()  # type: ignore

    if submissions:
        while submissions[0]["status"] == "pending":
            submissions = api.competitions_submissions_list(
                "tabular-playground-series-aug-2022", async_req=True
            ).get()  # type: ignore
            time.sleep(1)

        latest_submission = submissions[0]
        print("Date:", latest_submission["date"])
        print("Description:", latest_submission["description"])
        print("Status:", latest_submission["status"])
        print("Public Score:", latest_submission["publicScoreNullable"])
        print("Private Score:", latest_submission["privateScoreNullable"])
        return {
            "date": latest_submission["date"],
            "description": latest_submission["description"],
            "status": latest_submission["status"],
            "publicScore": latest_submission["publicScoreNullable"],
            "privateScore": latest_submission["privateScoreNullable"],
        }
    else:
        print("No submissions found.")
        return None


if __name__ == "__main__":
    test()
