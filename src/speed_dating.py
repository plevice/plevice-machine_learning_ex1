import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

from src.utils.random_imputer import RandomImputer
from src.utils.training import do_training

from preprocess import *
from classify import *

df = pd.read_csv("/home/christoph/TU/9_WS22/ML/plevice-machine_learning_ex1/data/raw/speeddating/speeddating_raw.csv")

X = df.copy().drop(["match"], axis=1)
y = df["match"].copy()

# remove all columns with > 5% missing values
# X.dropna(thresh=len(df) * 0.95, axis=1, inplace=True)

# remove intervall-ized information
X.drop([attribute for attribute in df.columns if attribute[:2] == "d_" and attribute != "d_age"], axis=1, inplace=True)

# why bother with an id?
X.drop("id", axis=1, inplace=True)

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
        ("prep", RandomImputer()),
        ("scaler", StandardScaler())
    ], numeric_features_zscale),
    ("en2c", [
        ("prep", RandomImputer()),
        ("scaler", MinMaxScaler())
    ], numeric_features_minmax),
    ("enc", [
        ("enc", OneHotEncoder(handle_unknown="ignore"))
    ], categorical_features),
]
for c in [
    BernoulliNB(),
    # MultinomialNB(),
    RandomForestClassifier(),
    DecisionTreeClassifier(max_depth=10),
    # RandomForestClassifier(min_samples_split=3),
    # KNeighborsClassifier(n_neighbors=3),
    KNeighborsClassifier(n_neighbors=4),
    KNeighborsClassifier(n_neighbors=5)
]:
    #  do_training(x, y, steps, c, cv=5)
    do_training(X, y, steps, c, test_size=0.2)

