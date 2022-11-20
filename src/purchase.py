from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

from src.utils.training import train_all_classifiers, do_prediction

from preprocess import *
from classify import *

df_train = pd.read_csv("data/raw/purchase/purchase600-100cls-15k.lrn.csv", sep=",")
df_test = pd.read_csv("data/raw/purchase/purchase600-100cls-15k.tes.csv", sep=",")

# train
var_columns = [c for c in df_train.columns if c not in ['ID', 'class']]

test_ids = df_test["ID"].copy()
# Create Train data on whole DataSet (needed for final modelfit and CV)
X_train = df_train.loc[:, var_columns].copy()
y_train = df_train.loc[:, 'class'].copy()
X_test = df_test.loc[:, var_columns].copy()

# Plot heatmap for correlation
# _, ax = plt.subplots(figsize=(31, 31))
# sns.heatmap(X_train.corr(), annot=True, linewidths=1, fmt='.2f', ax=ax)
# plt.show()


steps = [
]

train_all_classifiers(X_train, y_train, steps, multi=True, plot_y=(0, 0, 0), plot_y_max=(0.7, 0.7, 0.7))
train_all_classifiers(X_train, y_train, steps, multi=True, plot_y=(0, 0, 0), plot_y_max=(0.7, 0.7, 0.7), cv=5)

# for keggle
res_list = do_prediction(X_train, y_train, X_test, steps, MultinomialNB(alpha=2, fit_prior=True))
pd.DataFrame({"ID": test_ids, "class": res_list}).to_csv("data/raw/purchase/purchase600-100cls-15k.sol.ex.csv", index=False)
