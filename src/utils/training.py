from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, train_test_split
import pandas as pd

from src.utils.pipeline_debugger import Debugger


def do_training(x, y, preprocessing_steps, classifier, cv=None, test_size=0.2):
    if cv is not None and cv < 2:
        raise ValueError(f"cv {cv} is not valid!")

    trainer = DatasetTrainer()
    for step in preprocessing_steps:
        trainer.add_preprocessing_step(step[0], step[1], step[2])

    print("Training dataset with ",
          f"{classifier.__repr__()}, cv={cv}, test_size={test_size if cv is None else 1/cv}...", end="")
    if cv:
        scores = trainer.cross_validate(classifier, x, y, cv=cv)
        print(f"done.\nModel scores: {scores}")
    else:
        X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=test_size)
        trainer.train(classifier, X_train, y_train)
        print("Testing dataset...", end="")
        score = trainer.test(X_test, y_test)
        print("Done.")
        print("done. Model score: %.5f" % score)


class DatasetTrainer:

    def __init__(self):
        self.transformers = []
        self.clf = None

    def add_preprocessing_step(self, name, steps, columns):
        self.transformers.append((name, Pipeline(steps=steps), columns))

    def test(self, x_test, y_test):
        if self.clf is None:
            raise Exception("Called test before train!")

        return self.clf.score(x_test, y_test)

    def cross_validate(self, classifier, x, y, cv=10):
        self.__build_classifier(classifier)
        return cross_val_score(self.clf, x, y, cv=cv)

    def train(self, classifier, x_train, y_train):
        self.__build_classifier(classifier)
        return self.clf.fit(x_train, y_train)

    def __build_classifier(self, classifier):
        if len(self.transformers) == 0:
            raise Exception("No preprocessing steps added!")

        preprocessor = ColumnTransformer(
            transformers=self.transformers
        )

        self.clf = Pipeline(
            steps=[
                ("preprocessor", preprocessor),

                #      ("f", FunctionTransformer(lambda x: x.todense(), accept_sparse=True)),
                #  ("debug", Debugger()),
                ("classifier", classifier)
            ]
        )
