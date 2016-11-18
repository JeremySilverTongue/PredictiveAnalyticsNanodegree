import warnings
from collections import Counter

import numpy as np
import pandas as pd

warnings.filterwarnings(action="ignore",
                        module="scipy",
                        message="^internal gelsd")

TRAINING_FILENAME = "credit-data-training.xlsx"
TARGET_FILENAME = "customers-to-score.xlsx"

training = pd.read_excel(TRAINING_FILENAME)
target = pd.read_excel(TARGET_FILENAME)

y = training["Credit-Application-Result"]
training.drop(["Credit-Application-Result"], axis=1, inplace=True)

def clean_data(data):
    data = data.drop(["Occupation", "Concurrent-Credits", "Duration-in-Current-address"], axis=1)
    # data["Duration-in-Current-address"] = data["Duration-in-Current-address"].fillna(0)
    data["Age-years"] = data["Age-years"].fillna(data["Age-years"].mean())

    emp = {"< 1yr": 0, "1-4 yrs": 1, "4-7 yrs": 2}
    data["Length-of-current-employment"] = [emp[x] for x in data["Length-of-current-employment"]]

    value = {u"None": 0, u'< \xa3100': 1, u'\xa3100-\xa31000': 2}
    data["Value-Savings-Stocks"] = [value[x] for x in data["Value-Savings-Stocks"]]
    return data


training = clean_data(training)
target = clean_data(target)


print training.shape
print training["Age-years"].mean()

X = training = pd.get_dummies(training, drop_first=True)

print training.corr()

target = pd.get_dummies(target, drop_first=True)

for col in training:
    if col not in target:
        target[col] = 0

print training.shape
print target.shape

from sklearn.model_selection import cross_val_score


def test_classifier(clf, X, y):
    print clf.__class__, np.mean(cross_val_score(clf, X, y, cv=5, scoring="f1_macro"))


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

test_classifier(LogisticRegression(), X, y)
test_classifier(DecisionTreeClassifier(), X, y)
test_classifier(RandomForestClassifier(), X, y)
test_classifier(GradientBoostingClassifier(), X, y)

clf = LogisticRegression()

clf.fit(X, y)
print Counter(clf.predict(target))
