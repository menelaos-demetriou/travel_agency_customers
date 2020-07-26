import pickle
import pandas as pd
from utils import FeatureSelection

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score


def pre_process(data):
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

    # Drop unsused columns
    data = data.drop(columns=['CallStartTime', 'CallEndTime'])
    return data


def main(data):

    data = pre_process(data)

    # Split data to train validation and test set
    X_train, X_test, y_train, y_test = train_test_split(data.loc[:, ~data.columns.isin(['Outcome'])].copy(),
                                                        data["Outcome"].copy(), test_size=0.15,
                                                        random_state=18)

    num_attribs = ["Age", "ContactsTotal", "DaysFromPrevAttempt", "PrevAttempts", "CallDuration"]
    cat_attribs = ["Job", "MaritalStatus", "EducationLevel", "ContactMeans", "ContactMonth", "PrevOutcome"]
    ordinal_attribs = ["ContactDay"]

    preprocess = ColumnTransformer([("num", StandardScaler(), num_attribs),
                                    ("cat", OneHotEncoder(sparse=False), cat_attribs),
                                    ("ord", OrdinalEncoder(), ordinal_attribs)
                                    ])

    # List containing all hyperparameters for tuning
    search_space = [
        {"classifier": [LogisticRegression(max_iter=1000)],
         "classifier__penalty": ['l2'],
         "classifier__solver": ['newton-cg', 'saga', 'sag', 'liblinear', 'lbfgs']
         },
        {"classifier": [RandomForestClassifier()],
         "classifier__n_estimators": [10, 100, 1000],
         "classifier__max_depth": [5, 8, 15, 25, 30, None],
         "classifier__min_samples_leaf": [1, 2, 5, 10, 15, 100],
         "classifier__max_leaf_nodes": [2, 5, 10]
         }
    ]

    estimator = Pipeline([("preprocess", preprocess),
                          ("feature_selection", FeatureSelection()),
                          ("classifier", LogisticRegression())])

    # Performing grid search on all classifiers and hyperparameter combinations
    gridsearch = GridSearchCV(estimator, search_space, cv=5, verbose=1, scoring="f1", n_jobs=-1).fit(X_train, y_train)

    print("The best score is : %.2f" % gridsearch.best_score_)

    print("Saving best estimator for evaluations on test dataset")
    filename = "../models/best_estimator.sav"
    pickle.dump(gridsearch.best_estimator_, open(filename, 'wb'))

    print("Load model and perform evaluations")
    best_model = pickle.load(open(filename, "rb"))

    y_test_pred = best_model.predict(X_test)

    print("Accuracy is %.2f" % accuracy_score(y_test, y_test_pred))
    print("Precision is %.2f" % precision_score(y_test, y_test_pred))
    print("Recall is %.2f" % recall_score(y_test, y_test_pred))
    print("F1 score is %.2f" % f1_score(y_test, y_test_pred))

    result_dict = {"accuracy": accuracy_score(y_test, y_test_pred),
                   "precision": precision_score(y_test, y_test_pred),
                   "recall": recall_score(y_test, y_test_pred),
                   "f1_score": f1_score(y_test, y_test_pred)}

    results = pd.DataFrame([result_dict])
    results.to_csv("../results/best_model_metrics.csv", index=False)


if __name__ == "__main__":
    data = data = pd.read_csv("../data/campaign_data.csv")
    main(data)

