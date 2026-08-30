# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import pytest
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.utils.validation import check_is_fitted
from tryangle.model_selection.split import TriangleSplit
from tryangle.core.methods import CapeCod
from tryangle.metrics.score import neg_ave_scorer
from tryangle.utils.datasets import load_sample


def test_split():
    X = load_sample("swiss")
    triangle = X.triangle.copy()
    sample_weight = X.sample_weight.copy()
    tscv = TriangleSplit(n_splits=10)
    val_years = list(range(1988, 1998))

    for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
        assert (
            X[train_idx].triangle == triangle[triangle.valuation.year < val_years[i]]
        )
        assert (
            X[train_idx].sample_weight
            == sample_weight[sample_weight.origin.year < val_years[i]]
        )
        assert (
            X[test_idx].triangle == triangle[triangle.valuation.year <= val_years[i]]
        )


def test_split_no_leakage():
    """Train must not contain the diagonal being predicted (regression for #9)."""
    X = load_sample("swiss")
    triangle = X.triangle.copy()
    tscv = TriangleSplit(n_splits=10)
    val_years = list(range(1988, 1998))

    for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
        assert val_years[i] not in set(triangle.valuation[train_idx].year)
        assert set(train_idx) < set(test_idx)


def test_split_max_train_size():
    X = load_sample("swiss")
    triangle = X.triangle.copy()
    tscv = TriangleSplit(n_splits=10, max_train_size=3)
    val_years = list(range(1988, 1998))

    for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
        train_years = sorted(set(triangle.valuation[train_idx].year))
        assert len(train_years) <= 3
        assert max(train_years) == val_years[i] - 1
        assert (
            X[test_idx].triangle == triangle[triangle.valuation.year <= val_years[i]]
        )


def test_split_gap():
    X = load_sample("swiss")
    triangle = X.triangle.copy()
    tscv = TriangleSplit(n_splits=10, gap=2)
    val_years = list(range(1988, 1998))

    for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
        train_years = sorted(set(triangle.valuation[train_idx].year))
        assert max(train_years) == val_years[i] - 3
        assert (
            X[test_idx].triangle == triangle[triangle.valuation.year <= val_years[i]]
        )


@pytest.mark.parametrize("SearchClass", [(GridSearchCV), (RandomizedSearchCV)])
def test_search_methods(SearchClass):
    X = load_sample("swiss")
    tscv = TriangleSplit(n_splits=5)

    model = GridSearchCV(
        CapeCod(),
        param_grid={"decay": [0.2, 0.8]},
        scoring=neg_ave_scorer,
        cv=tscv,
        n_jobs=1,
    ).fit(X, X)
    assert check_is_fitted(model) is None
