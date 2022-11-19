from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

from src.utils.random_imputer import RandomImputer
from src.utils.training import do_training, plot_score, train_all_classifiers

from preprocess import *
from classify import *

df = pd.read_csv("data/raw/speeddating/speeddating_raw.csv")

X = df.copy().drop(["match"], axis=1)
y = df["match"].copy()

# remove all columns with > 5% missing values
# X.dropna(thresh=len(df) * 0.95, axis=1, inplace=True)

# remove intervall-ized information
X.drop([attribute for attribute in df.columns if attribute[:2] == "d_" and attribute != "d_age"], axis=1, inplace=True)

# why bother with an id?
X.drop("id", axis=1, inplace=True)

# 78.5% missing values -> no reliable data especially for cross-validation...
X = X.dropna(thresh=len(df)*0.15, axis=1).copy()

# ladies first
X["gender"] = X["gender"].map({"female": 0, "male": 1})

numeric_features_minmax = X.select_dtypes(
    include=np.number).columns.tolist()
# already scaled
numeric_features_minmax.remove("interests_correlate")
numeric_features_zscale = ["d_age", "age"]

for attr in numeric_features_zscale:
    numeric_features_minmax.remove(attr)

categorical_features = X.select_dtypes(include=['object']).columns.to_list()

steps = [
    ("num_z", [
        ("imputer", RandomImputer()),
        ("scaler", StandardScaler())
    ], numeric_features_zscale),
    ("minmax", [
        ("imputer", RandomImputer()),
        ("scaler", MinMaxScaler())
    ], numeric_features_minmax),
    ("enc", [
        ("enc", OneHotEncoder(handle_unknown="ignore"))
    ], categorical_features),
]


train_all_classifiers(X, y, steps)
