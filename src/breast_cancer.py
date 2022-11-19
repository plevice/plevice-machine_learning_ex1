import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

from src.utils.random_imputer import RandomImputer
from src.utils.training import do_training, do_prediction

from preprocess import *
from classify import *

df_train = pd.read_csv("data/raw/breast-cancer/breast-cancer-diagnostic.shuf.lrn.csv", sep=",")
df_test = pd.read_csv("data/raw/breast-cancer/breast-cancer-diagnostic.shuf.tes.csv", sep=",")


X_train = df_train.copy().drop(["class"], axis=1)
y_train = df_train["class"].copy()

X_test = df_test.copy()

test_ids = df_test["ID"].copy()

X_train.drop("ID", axis=1, inplace=True)
X_test.drop("ID", axis=1, inplace=True)

numeric_features_zscale = X_train.select_dtypes(
    include=np.number).columns.tolist()


steps = [
    ("num_z", [
        ("scaler", StandardScaler())
    ], numeric_features_zscale),
]

res_list = do_prediction(X_train, y_train, X_test, steps, RandomForestClassifier(n_estimators=10, criterion='entropy'))
pd.DataFrame({"ID": test_ids, "class": res_list}).to_csv("data/raw/breast-cancer/breast-cancer-diagnostic.shuf"
                                                         ".sol.ex.csv", index=False)

