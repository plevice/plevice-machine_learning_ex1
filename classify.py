# ---------------------------------------------------------------- #

# inspired by
# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

# ---------------------------------------------------------------- #

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB

# ---------------------------------------------------------------- #

def classify(X_train, y_train, X_test, y_test, parameters):

    classifiers_dict = {
        "KNN": [KNeighborsClassifier(k) for k in parameters["KNN"]],
        "DTC": [DecisionTreeClassifier(max_depth=max_depth) for max_depth in parameters["DTC"]],
        "RFC": [RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, max_features=max_features)
                for max_depth, n_estimators, max_features in parameters["RFC"]],
        "BNB": [BernoulliNB(alpha=alpha, binarize=binarize) for alpha, binarize in parameters["BNB"]],
        "MNB": [MultinomialNB(alpha=alpha) for alpha in parameters["BNB"]],
    }

    scores_dict = {"KNN": [], "DTC": [], "RFC": [], "BNB": [], "MNB": []}

    for kind, classifiers_list in classifiers_dict.items():
        for classifier in classifiers_list:

            classifier.fit(X_train, y_train)
            score = classifier.score(X_test, y_test)

            scores_dict[kind].append(score)

    return classifiers_dict, scores_dict

# ---------------------------------------------------------------- #
