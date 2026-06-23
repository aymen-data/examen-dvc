
import os

import pickle

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV



X_TRAIN_PATH = "data/processed/X_train_scaled.csv"

Y_TRAIN_PATH = "data/processed/y_train.csv"

OUT_PATH = "models/best_params.pkl"



def main():

    os.makedirs("models", exist_ok=True)



    X_train = pd.read_csv(X_TRAIN_PATH)

    y_train = pd.read_csv(Y_TRAIN_PATH).iloc[:, 0]



    model = RandomForestRegressor(random_state=42)



    param_grid = {

        "n_estimators": [50, 100],

        "max_depth": [None, 5, 10],

        "min_samples_split": [2, 5]

    }



    grid = GridSearchCV(

        estimator=model,

        param_grid=param_grid,

        cv=3,

        scoring="r2",

        n_jobs=-1

    )



    grid.fit(X_train, y_train)



    with open(OUT_PATH, "wb") as f:

        pickle.dump(grid.best_params_, f)



    print("GridSearch terminé.")

    print("Meilleurs paramètres :", grid.best_params_)

    print("Meilleur score R2 :", grid.best_score_)



if __name__ == "__main__":

    main()

