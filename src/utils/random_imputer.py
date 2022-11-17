from sklearn.base import BaseEstimator, TransformerMixin

from preprocess import random_imputer


class RandomImputer(BaseEstimator, TransformerMixin):

    def transform(self, data):
        df = data.copy()
        for attribute in data.columns:
            df[attribute] = random_imputer(data, attribute)

        return df

    def fit(self, data, y=None, **fit_params):
        return self
