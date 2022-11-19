import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.random_imputer import RandomImputer
from src.utils.training import do_training, do_prediction
from sklearn.feature_selection import RFECV

from preprocess import *
from classify import *

df_train = pd.read_csv("data/raw/breast-cancer/breast-cancer-diagnostic.shuf.lrn.csv", sep=",")
df_test = pd.read_csv("data/raw/breast-cancer/breast-cancer-diagnostic.shuf.tes.csv", sep=",")

# remove whitespaces
df_train.rename(columns=lambda x: x.strip(), inplace=True)
df_test.rename(columns=lambda x: x.strip(), inplace=True)

# train
X_train = df_train.copy().drop(["class"], axis=1)
y_train = df_train["class"].copy()

X_test = df_test.copy()
test_ids = df_test["ID"].copy()

X_test.drop("ID", axis=1, inplace=True)
X_train.drop("ID", axis=1, inplace=True)

# Plot heatmap for correlation
_, ax = plt.subplots(figsize=(31, 31))
sns.heatmap(X_train.corr(), annot=True, linewidths=1, fmt='.2f', ax=ax)
# plt.show()

# area, radius and perimeter are related -> drop 2
X_train.drop([col for col in X_train.columns if col.startswith("radius") or col.startswith("perimeter")], axis=1,
             inplace=True)
X_test.drop([col for col in X_train.columns if col.startswith("radius") or col.startswith("perimeter")], axis=1,
            inplace=True)

# concavity, concavePoints are related -> drop 1
X_train.drop([col for col in X_train.columns if col.startswith("concavePoints")], axis=1, inplace=True)
X_test.drop([col for col in X_train.columns if col.startswith("concavePoints")], axis=1, inplace=True)

# mean_cols = [col for col in df_train if col.endswith('mean')]
std_err_cols = [col for col in X_train if col.endswith('StdErr')]
X_train.drop(std_err_cols, axis=1, inplace=True)
X_test.drop(std_err_cols, axis=1, inplace=True)

# worst is already included in mean + stderr
# X_train.drop([col for col in X_train.columns if col.endswith('Worst')], axis=1, inplace=True)
# X_test.drop([col for col in X_test.columns if col.endswith('Worst')], axis=1, inplace=True)

classifier = RandomForestClassifier()
# The "accuracy" scoring is proportional to the number of correct classifications
# rfecv = RFECV(estimator=classifier, step=1, cv=10, scoring='accuracy')
# rfecv = rfecv.fit(X_train, y_train)
# best_f = X_train.columns[rfecv.support_]

# print('Best features :', best_f)

numeric_features_zscale = X_train.select_dtypes(
    include=np.number).columns.tolist()

steps = [
    ("num_z", [
        ("scaler", StandardScaler())
    ], numeric_features_zscale),
]

res_list = do_prediction(X_train, y_train, X_test, steps, classifier)
pd.DataFrame({"ID": test_ids, "class": res_list}).to_csv("data/raw/breast-cancer/breast-cancer-diagnostic.shuf"
                                                         ".sol.ex.csv", index=False)
