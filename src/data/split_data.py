
import os

import pandas as pd

from sklearn.model_selection import train_test_split



RAW_PATH = "data/raw/raw.csv"

OUT_DIR = "data/processed"

TARGET = "silica_concentrate"



def main():

    os.makedirs(OUT_DIR, exist_ok=True)



    df = pd.read_csv(RAW_PATH)



    X = df.drop(columns=[TARGET])

    y = df[TARGET]



    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.2,

        random_state=42

    )



    X_train.to_csv(f"{OUT_DIR}/X_train.csv", index=False)

    X_test.to_csv(f"{OUT_DIR}/X_test.csv", index=False)

    y_train.to_csv(f"{OUT_DIR}/y_train.csv", index=False)

    y_test.to_csv(f"{OUT_DIR}/y_test.csv", index=False)



    print("Split des données terminé.")



if __name__ == "__main__":

    main()

