from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

from src.utils.random_imputer import RandomImputer
from src.utils.training import do_training, plot_score, get_classifiers, train_all_classifiers

from preprocess import *
from classify import *

df = pd.read_csv("data/raw/mushroom/mushrooms_raw.csv", delimiter=";")

X_raw = df.copy().drop(["class"], axis=1)
y_raw = df["class"].copy()
# %%
X = X_raw.dropna(thresh=len(df) * 0.80, axis=1).copy()

for attribute in ["does-bruise-or-bleed", "has-ring"]:
    X[attribute] = X[attribute].map({"t": True, "f": False})

X["season"] = X["season"].map({"s": 0, "u": 1, "a": 2, "w": 3})

y = y_raw.map({"e": True, "p": False}).copy()

steps = [
    ("num_z", [
        ("imputer", RandomImputer()),
        ("scaler", StandardScaler())
    ], ["cap-diameter", "stem-height", "stem-width"]),
    ("enc", [
        ("enc", OneHotEncoder(handle_unknown="ignore"))
    ], [attribute for attribute, dtype in dict(X.dtypes).items() if dtype is np.dtype("O")]),
]

train_all_classifiers(X, y, steps, (0.75, 0.72, 0.74))
