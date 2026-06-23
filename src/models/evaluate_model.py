
import os

import json

import joblib

import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



X_TEST_PATH = "data/processed/X_test_scaled.csv"

Y_TEST_PATH = "data/processed/y_test.csv"

MODEL_PATH = "models/trained_model.pkl"

SCORES_PATH = "metrics/scores.json"

PREDICTIONS_PATH = "data/predictions.csv"



def main():

    os.makedirs("metrics", exist_ok=True)

    os.makedirs("data", exist_ok=True)



    X_test = pd.read_csv(X_TEST_PATH)

    y_test = pd.read_csv(Y_TEST_PATH).iloc[:, 0]



    model = joblib.load(MODEL_PATH)



    y_pred = model.predict(X_test)



    mse = mean_squared_error(y_test, y_pred)

    rmse = mse ** 0.5

    mae = mean_absolute_error(y_test, y_pred)

    r2 = r2_score(y_test, y_pred)



    scores = {

        "mse": float(mse),

        "rmse": float(rmse),

        "mae": float(mae),

        "r2": float(r2)

    }



    with open(SCORES_PATH, "w", encoding="utf-8") as f:

        json.dump(scores, f, indent=4)



    predictions = pd.DataFrame({

        "y_true": y_test,

        "y_pred": y_pred

    })

    predictions.to_csv(PREDICTIONS_PATH, index=False)



    print("Évaluation terminée.")

    print("Scores :", scores)



if __name__ == "__main__":

    main()

