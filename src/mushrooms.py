from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.utils.random_imputer import RandomImputer
from src.utils.training import train_all_classifiers

from preprocess import *

df = pd.read_csv("data/raw/mushroom/mushrooms_raw.csv", delimiter=";")

# there is only 'u' or NaN
df = df.drop(["veil-type"], axis=1)

X_raw = df.copy().drop(["class"], axis=1)
y_raw = df["class"].copy()
# %%
X = X_raw.dropna(thresh=len(df) * 0.8, axis=1).copy()

for attribute in ["does-bruise-or-bleed", "has-ring"]:
    X[attribute] = X[attribute].map({"t": True, "f": False})

X["season"] = X["season"].map({"s": 0, "u": 1, "a": 2, "w": 3})

y = y_raw.map({"e": True, "p": False}).copy()

steps = [
    ("num_z", [
        ("imputer", RandomImputer()),
        ("scaler", StandardScaler())
    ], ["cap-diameter", "stem-height", "stem-width"]),
    ("rimp", [
        ("imputer", RandomImputer()),
        ("enc", OneHotEncoder(handle_unknown="ignore"))
    ], ["gill-attachment", "ring-type"]),
    ("enc", [
        # ("imputer", RandomImputer()),
        ("enc", OneHotEncoder(handle_unknown="ignore"))
    ], [attribute for attribute, dtype in dict(X.dtypes).items() if dtype is np.dtype("O") and attribute not in ["gill-attachment", "ring-type"]]),
]

train_all_classifiers(X, y, steps, (0.5, 0.5, 0.5), cv=5)
