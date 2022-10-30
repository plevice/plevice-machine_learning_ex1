from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score


def do_training(x_train, y_train, x_test, y_test, preprocessing_steps, classifier):
    trainer = DatasetTrainer()
    for step in preprocessing_steps:
        trainer.add_preprocessing_step(step[0], step[1], step[2])

    print("Training dataset with {}...".format(classifier.__repr__()), end="")
    trainer.train(classifier, x_train, y_train)
    print("Done.")
    print("Testing dataset...", end="")
    score = trainer.test(x_test, y_test)
    print("done. Model score: %.3f" % score)


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

    def train(self, classifier, x_train, y_train):
        if len(self.transformers) == 0:
            raise Exception("No preprocessing steps added!")

        preprocessor = ColumnTransformer(
            transformers=self.transformers
        )

        self.clf = Pipeline(
            steps=[
                ("preprocessor", preprocessor),

          #      ("f", FunctionTransformer(lambda x: x.todense(), accept_sparse=True)),
                #      ("debug", Debugger()),
                ("classifier", classifier)
            ]
        )

       #  print(cross_val_score(self.clf, x_train, y_train, cv=10))

        return self.clf.fit(x_train, y_train)
