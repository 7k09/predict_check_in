import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier


from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
import seaborn as sns
import numpy as np
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import AdaBoostClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.metrics.classification import recall_score
from sklearn import preprocessing
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import preprocessing
from patsy import dmatrices
from sklearn import cross_validation

def feature_importance(X,y):
    # Build a forest and compute the feature importances
#     forest = ExtraTreesClassifier(n_estimators=50,
#                               random_state=2)
    forest = RandomForestClassifier(n_estimators=50)
    X = X.ix[:,1:]

    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
    indices = np.argsort(importances)[::]
    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    fig = plt.figure(figsize=(12,8))
    plt.title("Reconnect Model Feature importances",fontsize=20)
    plt.barh(range(X.shape[1]), importances[indices],
       color="#ffc61e", xerr=std[indices], align="center",alpha=0.8)
    x_labels = [X.columns[i] for i in  indices]
    plt.yticks(range(X.shape[1]),x_labels,fontsize=15)
    plt.ylim([-1, X.shape[1]])

    plt.show()
def log_reg(X,y):
    logit = sm.Logit(y, X)
    # fit the model
    result = logit.fit()
    print result.summary()
    # odds ratios only
    print np.exp(result.params)
    return

def model_evaluation_nonweight(X,y, model=LogisticRegression(),k=5, sample_weight=None):
    # instantiate a logistic regression model, and fit with X and y
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state=5)

    model = model.fit(X_train, y_train)
    result = model.predict(X_test)
    accuracy = accuracy_score(y_test,result)
    cnt = 0.0
    for i,j in zip(y_test, result):
        if str(i).strip() ==str(j).strip():
            cnt+=1
    print cnt/len(y_test)
    f1 = f1_score(y_test,result)
    precision = precision_score(y_test,result)
    recall = recall_score(y_test,result)
    print accuracy,f1,precision,recall

def model_evaluation(X,y, model=LogisticRegression(),k=5, sample_weight=None):
#     X['weight'] = sample_weight
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state=5)
    train_weight = X_train.weight.values
    test_weight = X_test.weight.values
    X_train.drop('weight',1)
    X_test.drop('weight',1)
    model = model.fit(X_train, y_train, train_weight)
    result = model.predict(X_test)
    accuracy = accuracy_score(y_test,result,sample_weight=test_weight)

    f1 = f1_score(y_test,result,sample_weight=test_weight)

    precision = precision_score(y_test,result,sample_weight=test_weight)

    recall = recall_score(y_test,result,sample_weight=test_weight)

    print accuracy,f1,precision,recall

def model_evaluation_cross(X,y, model=LogisticRegression(),k=5, fps = None):
    accuracy = cross_val_score(model, X, y, scoring='accuracy', cv=k, fit_params = fps)
    f1 = cross_val_score(model, X, y, scoring='f1', cv=k, fit_params = fps)
    precision =cross_val_score(model, X, y, scoring='precision', cv=k, fit_params = fps)
    recall =cross_val_score(model, X, y, scoring='recall', cv=k, fit_params = fps)
#     roc = cross_val_score(model, X, y, scoring='roc_auc', cv=k)
    print accuracy.mean(),f1.mean(),recall.mean(),precision.mean()


def check_vif(X):
    X_test = X.ix[:,1:]

    for i in range(len(X_test.columns)):
        print variance_inflation_factor(np.array(X_test),i)
