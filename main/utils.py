import pickle
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin


class Utilities(object):

    @staticmethod
    def get_column_names_from_ColumnTransformer(column_transformer):
        col_name = []

        for transformer_in_columns in column_transformer.transformers_[
                                      :-1]:

            raw_col_name = list(transformer_in_columns[2])

            if isinstance(transformer_in_columns[1], Pipeline):
                # if pipeline, get the last transformer
                transformer = transformer_in_columns[1].steps[-1][1]
            else:
                transformer = transformer_in_columns[1]

            try:
                if isinstance(transformer, OneHotEncoder):
                    names = list(transformer.get_feature_names(raw_col_name))

                elif isinstance(transformer, SimpleImputer) and transformer.add_indicator:
                    missing_indicator_indices = transformer.indicator_.features_
                    missing_indicators = [raw_col_name[idx] + '_missing_flag' for idx in missing_indicator_indices]

                    names = raw_col_name + missing_indicators

                else:
                    names = list(transformer.get_feature_names())

            except AttributeError as error:
                names = raw_col_name

            col_name.extend(names)

        return col_name


class FeatureSelection(BaseEstimator, TransformerMixin):

    def __init__(self):
        with open("../models/optimal_features.txt", "rb") as fp:
            self.indexes = pickle.load(fp)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X[:, self.indexes]
        return X


