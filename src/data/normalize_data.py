
import pandas as pd

from sklearn.preprocessing import StandardScaler



IN_DIR = "data/processed"



def main():

    X_train = pd.read_csv(f"{IN_DIR}/X_train.csv")

    X_test = pd.read_csv(f"{IN_DIR}/X_test.csv")



    # On garde uniquement les colonnes numériques

    X_train = X_train.select_dtypes(include=["number"])

    X_test = X_test[X_train.columns]



    scaler = StandardScaler()



    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)



    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)

    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)



    X_train_scaled.to_csv(f"{IN_DIR}/X_train_scaled.csv", index=False)

    X_test_scaled.to_csv(f"{IN_DIR}/X_test_scaled.csv", index=False)



    print("Normalisation terminée.")



if __name__ == "__main__":

    main()

