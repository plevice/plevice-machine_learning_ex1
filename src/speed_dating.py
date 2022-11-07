import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, Normalizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from classify import classify
from preprocess import preprocess
from src.utils.training import do_training


def main():
    print("Fetching dataset...", end="")
    dataset = fetch_openml(data_id=40536,
                           as_frame=True)
    print("done")

    df = dataset.frame
    del df['has_null']
    # 78.5% missing values -> keep or not?
    del df["expected_num_interested_in_me"]
    convert_categories_to_boolean(df, ["samerace", "match"])

    # handle missing numeric values
    numeric_features_minmax = df.select_dtypes(
        include=np.number).columns.tolist()
    # already scaled
    numeric_features_minmax.remove("interests_correlate")
    numeric_features_zscale = ["d_age", "age"]

    for attr in numeric_features_zscale:
        numeric_features_minmax.remove(attr)

    x = dataset.data
    y = dataset.target
    categorical_features = df.select_dtypes(
        include=['category']).columns.to_list()

    steps = [
        ("num_z", [
            ("imputer", KNNImputer(n_neighbors=1)),
            ("scaler", StandardScaler())
        ], numeric_features_zscale),

        ("num_min_max", [
            ("imputer", KNNImputer(n_neighbors=1)),
            ("scaler", MinMaxScaler())
        ], numeric_features_minmax),

        ("cat", [
            ("encode", OneHotEncoder(handle_unknown="ignore"))
        ], categorical_features),
    ]

    print(f"Preprocessing steps are:")
    for step in steps:
        print(f"{step[0]} -> {step[1]}")

    for c in [
        # BernoulliNB(),
        # MultinomialNB(),
        # RandomForestClassifier(),
        DecisionTreeClassifier(max_depth=10),
        RandomForestClassifier(min_samples_split=3),
        KNeighborsClassifier(n_neighbors=4)
    ]:
        do_training(x, y, steps, c, cv=5)
        # do_training(x, y, steps, c, test_size=0.1)


#  print(scores_dict)


def convert_categories_to_boolean(df, attrs):
    for attr in attrs:
        df[attr] = df[attr].map({"0": False, "1": True}).astype("bool")


main()
