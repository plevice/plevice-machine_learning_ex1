# ---------------------------------------------------------------- #
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

pd.set_option('display.max_columns', None)

import random
from scipy.stats import zscore


# from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------- #

def missing_values(df):
    df_isnull_abs = df.isnull().sum(axis=0)
    df_isnull_rel = round(df_isnull_abs / len(df) * 100, 2)

    return pd.concat([df_isnull_abs, df_isnull_rel], axis=1).rename(columns={0: "absolute", 1: "relative"})


# -------------------------------- #

def random_imputer(df, attribute):
    """
    We replace all missing values of an attribute by a random value.
    The latter is selected by sampling from the probability distribution induced by the occurences of the existing values.
    """

    column = df[attribute].copy()
    column_full = column.loc[~pd.isnull(column)]
    column_empty = column.loc[pd.isnull(column)]

    if len(column_full) == 0:
        raise ValueError("No data to sample from!")

    try:
        sample = pd.Series(
            random.choices(
                population=list(column_full.value_counts().index),
                weights=list(column_full.value_counts() / len(column_full)),
                k=len(column_empty)
            )
        )
        sample.index = column.loc[pd.isnull(column)].index
        column.loc[pd.isnull(column)] = sample
    except:
        print(column_full)
        print(column_empty)

    return column


# -------------------------------- #

def preprocess(X, y, attributes, categories):
    # handle missing values
    df_nan = pd.concat([X, y], axis=1)
    for attribute in X.columns:
        df_nan[attribute] = random_imputer(X, attribute)
    X_nan, y_nan = df_nan.iloc[:, :-1], df_nan.iloc[:, -1]

    # one hot encoding
    X_ohe = pd.concat(
        [
            pd.get_dummies(X_nan[attribute].astype(pd.CategoricalDtype(categories=categories[attribute])))
            for attribute in attributes["ohe"]
        ],
        axis=1
    )

    # z scaling
    X_zsc = X_nan[attributes["zsc"]].apply(zscore)

    # collect the rest
    X_rest = X_nan[[
        attribute for attribute in X_nan.columns
        if not (attribute in attributes["ohe"] or attribute in attributes["zsc"])
    ]]

    # concatenate
    X_preprocessed = pd.concat([X_ohe, X_zsc, X_rest], axis=1)
    y_preprocessed = y_nan

    return X_preprocessed, y
