from typing import Iterator, Optional

from sklearn.model_selection import KFold


class KFoldCrossValidation:
    """
    Split the data in k-folds.
    While the original sklearn.model_selection.KFold returns the index in the iteration,
    an instance of this class returns the objects instead the indexes.
    """

    def __init__(self, X, y, n_splits: int, random_state: Optional[int]=None):
        self.X = X
        self.y = y
        self._iterator = None
       
        self.kfolds = [
            KFold(n_splits=n_splits, random_state=random_state, shuffle=random_state is not None),
            KFold(n_splits=n_splits, random_state=random_state, shuffle=random_state is not None)
        ]

    def split(self):
        self._iterator = enumerate(
            zip(
                self.kfolds[0].split(self.X),
                self.kfolds[1].split(self.y)
            )
        )
        return self

    def __iter__(self) -> Iterator:
        return self

    def __next__(self):
        index, ((X_train_index, X_test_index), (y_train_index, y_test_index)) = self._iterator.__next__()
        X_train = self.X[X_train_index]
        X_test = self.X[X_test_index]

        y_train = self.y[y_train_index]
        y_test = self.y[y_test_index]

        return index, X_train, X_test, y_train, y_test
