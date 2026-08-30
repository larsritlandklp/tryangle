# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
from sklearn.model_selection._split import TimeSeriesSplit


class TriangleSplit(TimeSeriesSplit):
    """Triangle cross-validation across calendar periods

    Splits a ``TryangleData`` instance into k sub-triangles or folds
    over the latest k calendar periods or diagonals. Thus, successive
    folds are supersets of previous folds, unless ``max_train_size``
    is set, in which case each training fold is capped to that many
    diagonals (rolling window instead of expanding window).

    Parameters
    ----------
    n_splits : int, default=5
        Number of splits, must be at least 2.
    max_train_size : int, default=None
        Maximum number of diagonals in a training fold. If None,
        folds use an expanding window (all diagonals up to the
        test point).
    gap : int, default=0
        Number of diagonals to exclude between train and test.
    """

    def __init__(self, n_splits=5, *, max_train_size=None, gap=0):
        super().__init__(
            n_splits=n_splits, max_train_size=max_train_size, test_size=1, gap=gap
        )

    def split(self, X, y=None, groups=None):
        valuation_date = X.triangle.latest_diagonal.valuation[0]
        valuation_dates = np.array(
            [
                date
                for date in X.triangle.valuation.drop_duplicates().sort_values()
                if date.date() <= valuation_date.date()
            ]
        )
        for train, test in super().split(valuation_dates):
            train_start_date = valuation_dates[train[0]]
            train_end_date = valuation_dates[train[-1]]
            test_date = valuation_dates[test[0]]
            yield (
                np.where(
                    (X.triangle.valuation >= train_start_date)
                    & (X.triangle.valuation <= train_end_date)
                )[0],
                np.where(X.triangle.valuation <= test_date)[0],
            )
