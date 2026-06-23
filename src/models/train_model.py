
import os

import pickle

import joblib

import pandas as pd

from sklearn.ensemble import RandomForestRegressor



X_TRAIN_PATH = "data/processed/X_train_scaled.csv"

Y_TRAIN_PATH = "data/processed/y_train.csv"

PARAMS_PATH = "models/best_params.pkl"

MODEL_PATH = "models/trained_model.pkl"



def main():

    os.makedirs("models", exist_ok=True)



    X_train = pd.read_csv(X_TRAIN_PATH)

    y_train = pd.read_csv(Y_TRAIN_PATH).iloc[:, 0]



    with open(PARAMS_PATH, "rb") as f:

        best_params = pickle.load(f)



    model = RandomForestRegressor(

        random_state=42,

        **best_params

    )



    model.fit(X_train, y_train)



    joblib.dump(model, MODEL_PATH)



    print("Entraînement terminé.")

    print("Modèle sauvegardé dans :", MODEL_PATH)



if __name__ == "__main__":

    main()

