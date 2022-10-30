import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from src.utils.training import do_training


def main():
    print("Fetching dataset...", end="")
    dataset = fetch_openml(data_id=40536,
                           as_frame=True)
    print("done")

    df = dataset.frame
    del df['has_null']
    convert_categories_to_boolean(df, ["samerace", "match"])

    # handle missing numeric values
    numeric_features_minmax = df.select_dtypes(
        include=np.number).columns.tolist()
    numeric_features_zscale = ["d_age", "age"]

    for attr in numeric_features_zscale:
        numeric_features_minmax.remove(attr)

    x = dataset.data
    y = dataset.target
    categorical_features = df.select_dtypes(
        include=['category']).columns.to_list()

    steps = [
        ("num_z", [
            ("imputer", KNNImputer(n_neighbors=8)),
            ("scaler", StandardScaler())
        ], numeric_features_zscale),

        ("num_min_max", [
            ("imputer", KNNImputer(n_neighbors=8)),
            ("scaler", MinMaxScaler())
        ], numeric_features_minmax),

        ("cat", [
            #        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encode", OneHotEncoder(handle_unknown="ignore"))
        ], categorical_features),
    ]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=10)
    for c in [
        RandomForestClassifier(),
        # RandomForestClassifier(min_samples_split=3),
    ]:
        do_training(x_train, y_train, x_test, y_test, steps, c)


def convert_categories_to_boolean(df, attrs):
    for attr in attrs:
        df[attr] = df[attr].map({"0": False, "1": True}).astype("bool")


main()
