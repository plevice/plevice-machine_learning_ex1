from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

from src.utils.pipeline_debugger import Debugger


def do_training(x, y, preprocessing_steps, classifier, cv=None, test_size=0.2):
    if cv is not None and cv < 2:
        raise ValueError(f"cv {cv} is not valid!")

    trainer = DatasetTrainer(classifier.__repr__())
    for step in preprocessing_steps:
        trainer.add_preprocessing_step(step[0], step[1], step[2])

    print("Training dataset with ",
          f"{classifier.__repr__()}, cv={cv}, test_size={test_size if cv is None else ''}...", end="")
    if cv:
        scores = trainer.cross_validate(classifier, x, y, cv=cv)
        print(f"done.\nModel scores: {scores}")
    else:
        X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=12, test_size=test_size)

        trainer.train(classifier, X_train, y_train)
        print("Testing dataset...", end="")
        score = trainer.test(X_test, y_test)
        print("Done.")
        print(f"done. Model scores: {score}")
        return score


def do_prediction(X_train, y_train, X_test, preprocessing_steps, classifier):
    trainer = DatasetTrainer(classifier.__repr__())
    for step in preprocessing_steps:
        trainer.add_preprocessing_step(step[0], step[1], step[2])

    print("Training dataset with ",
          f"{classifier.__repr__()}...", end="")

    trainer.train(classifier, X_train, y_train)
    print("Predicting dataset...", end="")
    res = trainer.predict(X_test)
    print("Done.")
    return res


class DatasetTrainer:
    name = ""

    def __init__(self, name=""):
        self.transformers = []
        self.clf = None
        self.name = name

    def add_preprocessing_step(self, name, steps, columns):
        self.transformers.append((name, Pipeline(steps=steps), columns))

    def test(self, x_test, y_test):
        if self.clf is None:
            raise Exception("Called test before train!")

        plt.figure(figsize=(7, 4))
        plt.suptitle(self.name)
        cm = confusion_matrix(y_test, self.clf.predict(x_test))

        tp = cm[0][0]
        fp = cm[0][1]
        fn = cm[1][0]
        tn = cm[1][1]
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * ((precision * recall) / (precision + recall))

        sns.heatmap(cm, annot=True, fmt="d")
        plt.show()

        return precision, recall, f1

    def predict(self, x_test):
        if self.clf is None:
            raise Exception("Called test before train!")

        return self.clf.predict(x_test)

    def cross_validate(self, classifier, x, y, cv=10):
        self.__build_classifier(classifier)
        return cross_val_score(self.clf, x, y, cv=cv, error_score="raise")

    def train(self, classifier, x_train, y_train):
        self.__build_classifier(classifier)
        return self.clf.fit(x_train, y_train)

    def __build_classifier(self, classifier):
        #      if len(self.transformers) == 0:
        #        raise Exception("No preprocessing steps added!")

        if len(self.transformers) > 0:
            self.clf = Pipeline(
                steps=[
                    ("preprocessor", ColumnTransformer(transformers=self.transformers)),

                    #      ("f", FunctionTransformer(lambda x: x.todense(), accept_sparse=True)),
                    #  ("debug", Debugger()),
                    ("classifier", classifier)
                ]
            )
        else:
            self.clf = classifier


def plot_score(plot_data, title, ylim=(0.85, 1)):
    names = list(plot_data.keys())
    values = list(plot_data.values())
    print(plot_data)

    plt.figure(figsize=(8, 3))
    plt.ylim(ylim)
    plt.bar(names, values)
    plt.suptitle(title)
    plt.show()



def train_all_classifiers(X, y, steps, plot_y=(0.85, 0.85, 0.85)):
    plot_data_prec = {}
    plot_data_rec = {}
    plot_data_f1 = {}
    i = 0
    for c in get_classifiers():
        name = c.__repr__()
        # do_training(X, y, steps, c, cv=5)
        scores = do_training(X, y, steps, c, test_size=0.2)
        plot_data_prec[i] = scores[0]
        plot_data_rec[i] = scores[1]
        plot_data_f1[i] = scores[2]
        i = i + 1
    plot_score(plot_data_prec, "Precision", (plot_y[0], 1))
    plot_score(plot_data_rec, "Recall", (plot_y[1], 1))
    plot_score(plot_data_f1, "F1 Score", (plot_y[2], 1))

def get_classifiers():
    return [
        KNeighborsClassifier(n_neighbors=1, weights='uniform'),
        KNeighborsClassifier(n_neighbors=5, weights='uniform'),
        KNeighborsClassifier(n_neighbors=10, weights='uniform'),
        KNeighborsClassifier(n_neighbors=50, weights='uniform'),

        KNeighborsClassifier(n_neighbors=1, weights='distance', metric='minkowski'),
        KNeighborsClassifier(n_neighbors=5, weights='distance', metric='minkowski'),
        KNeighborsClassifier(n_neighbors=10, weights='distance', metric='minkowski'),
        KNeighborsClassifier(n_neighbors=50, weights='distance', metric='minkowski'),

        KNeighborsClassifier(n_neighbors=1, weights='distance', metric='cityblock'),
        KNeighborsClassifier(n_neighbors=5, weights='distance', metric='cityblock'),
        KNeighborsClassifier(n_neighbors=10, weights='distance', metric='cityblock'),
        KNeighborsClassifier(n_neighbors=50, weights='distance', metric='cityblock'),

        BernoulliNB(alpha=0),
        BernoulliNB(alpha=1),
        BernoulliNB(alpha=10),

        DecisionTreeClassifier(max_depth=10, criterion='gini'),
        DecisionTreeClassifier(max_depth=20, criterion='gini'),
        DecisionTreeClassifier(max_depth=30, criterion='gini'),
        DecisionTreeClassifier(max_depth=10, criterion='entropy'),
        DecisionTreeClassifier(max_depth=20, criterion='entropy'),
        DecisionTreeClassifier(max_depth=30, criterion='entropy'),
    ]
