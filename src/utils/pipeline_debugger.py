from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd
import time

from preprocess import preprocess


class Debugger(BaseEstimator, TransformerMixin):

    def transform(self, data):
        print("Shape of Pre-processed Data:", data.shape)
        df = pd.DataFrame(data)
        print(df.head())
       # df.to_csv("debug/{}.csv".format(time.time()))
        return data

    def fit(self, data, y=None, **fit_params):
        return self
