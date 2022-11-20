import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.random_imputer import RandomImputer
from src.utils.training import do_training, do_prediction, train_all_classifiers
from sklearn.feature_selection import RFECV

from preprocess import *
from classify import *

df_train = pd.read_csv("data/raw/purchase/purchase600-100cls-15k.lrn.csv", sep=",")
df_test = pd.read_csv("data/raw/purchase/purchase600-100cls-15k.tes.csv", sep=",")

# train
var_columns = [c for c in df_train.columns if c not in ['ID', 'class']]

# Create Train data on whole DataSet (needed for final modelfit and CV)
X_train = df_train.loc[:, var_columns].copy()
y_train = df_train.loc[:, 'class'].copy()
X_test = df_test.loc[:, var_columns].copy()

# Plot heatmap for correlation
# _, ax = plt.subplots(figsize=(31, 31))
# sns.heatmap(X_train.corr(), annot=True, linewidths=1, fmt='.2f', ax=ax)
# plt.show()


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

train_all_classifiers(X_train, y_train, steps, cv=2, multi=True)
