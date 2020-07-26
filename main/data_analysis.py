import pickle
import pandas as pd
import seaborn as sns
from utils import Utilities
from itertools import compress
import matplotlib.pyplot as plt
from dython.nominal import associations

from sklearn.feature_selection import RFECV
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
plt.style.use('ggplot')


def plot_distribution(data, feature, numeric_flag):
    if numeric_flag:
        sns.boxplot(y=feature, x="Outcome", data=data)
        plt.xlabel("Outcome")
        plt.ylabel(feature)
        plt.title("%s Distribution" % feature)
    else:
        sns.countplot(x=feature, hue="Outcome", data=data)
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.title("%s Distribution" % feature)
        plt.xticks(rotation=45)
        plt.tight_layout()
    plt.savefig("../plots/%s_distribution" % feature)
    plt.show()


def pre_process(data):
    # Replace null with appropriate values
    data["Job"] = data["Job"].fillna("not_known")
    data["EducationLevel"] = data["EducationLevel"].fillna("not_known")
    data["ContactMeans"] = data["ContactMeans"].fillna("not_known")
    data["PrevOutcome"] = data["PrevOutcome"].fillna("not_known")
    data["DaysFromPrevAttempt"] = data["DaysFromPrevAttempt"].replace(-1, 0)

    # convert the call columns to datetime format
    data['CallStartTime'] = pd.to_datetime(data['CallStartTime'])
    data['CallEndTime'] = pd.to_datetime(data['CallEndTime'])
    data["CallDuration"] = data["CallEndTime"] - data["CallStartTime"]
    data["CallDuration"] = data["CallDuration"].dt.seconds

    month_dict = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9,
                  "oct": 10, "nov": 11, "dec": 12}

    data["ContactMonth"] = data["ContactMonth"].replace(month_dict)
    # Drop unsused columns
    data = data.drop(columns=['CallStartTime', 'CallEndTime'])
    return data


def plotting(data):
    # Plot each feature with target
    numeric_features = ["Age", "DaysFromPrevAttempt", "PrevAttempts", "CallDuration", "ContactsTotal"]
    features = list(data.columns)
    features.remove("Outcome")
    for feature in features:
        if feature in numeric_features:
            numeric_flag = True
        else:
            numeric_flag = False

        plot_distribution(data, feature, numeric_flag)


def correlations(data):
    associations(data, figsize=(15, 15), cmap="viridis")
    plt.show()


def feature_selection(data):

    clf = LogisticRegression(max_iter=1000)
    y = data["Outcome"].copy()
    X = data.loc[:, ~data.columns.isin(['Outcome'])].copy()

    num_attribs = ["Age", "ContactsTotal", "DaysFromPrevAttempt", "PrevAttempts", "CallDuration"]
    cat_attribs = ["Job", "MaritalStatus", "EducationLevel", "ContactMeans", "ContactMonth", "PrevOutcome"]
    ordinal_attribs = ["ContactDay"]

    full_pipeline = ColumnTransformer([("num", StandardScaler(), num_attribs),
                                       ("cat", OneHotEncoder(sparse=False), cat_attribs),
                                       ("ord", OrdinalEncoder(), ordinal_attribs)
                                       ])

    X_p = full_pipeline.fit_transform(X)
    rfecv = RFECV(estimator=clf, step=1, cv=StratifiedKFold(2),
                  scoring='f1')
    rfecv.fit(X_p, y)

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.savefig("../plots/RFE.png")
    plt.show()

    print("Optimal number of features : %d" % rfecv.n_features_)
    get_names = Utilities()
    feature_names = get_names.get_column_names_from_ColumnTransformer(full_pipeline)

    res = list(compress(feature_names, rfecv.support_))
    non_res = list(compress(feature_names, ~rfecv.support_))
    print("The list with the optimal features is: ", str(res))
    print("The list with non optimal features is: ", str(non_res))

    indexes = [i for i, val in enumerate(rfecv.support_) if val]
    with open("../models/optimal_features.txt", "wb") as fp:
        pickle.dump(indexes, fp)


def main():
    # Read csv file with dataset
    data = pd.read_csv("../data/campaign_data.csv")

    # Data type of each feature
    print(data.info())

    # Check Null entries
    print(round(100*(data.isnull().sum()/len(data.index)), 2))

    # Preprocess before data analysis
    data = pre_process(data)

    # Check if imbalanced test set
    print(data["Outcome"].value_counts(normalize=True))

    # Plot all features against the target
    plotting(data)

    # Check correlations between features
    correlations(data)

    # Get most important features
    feature_selection(data)


if __name__ == "__main__":
    main()

