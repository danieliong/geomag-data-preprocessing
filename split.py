#############################################################################
# split.py
#
# Description: Contains functions/classess for splitting data into training/testing.
# Options include: time-series split (sequential split) or storm split
#############################################################################

import logging
import re
from collections import namedtuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from utils import is_pandas
from hydra.utils import to_absolute_path

from storm_utils import StormAccessor, StormIndexAccessor

logger = logging.getLogger(__name__)


TRAIN = namedtuple("TRAIN", ["X", "y"])
TEST = namedtuple("TEST", ["X", "y"])


class StormSplitter:
    def __init__(self, times_path, seed=None):
        self.storm_times = pd.read_csv(times_path, index_col=0)
        self.rng = np.random.default_rng(seed)

    @property
    def storms(self):
        return self.storm_times.index

    @property
    def n_storms(self):
        return len(self.storms)

    def _storm_iter(self):
        return self.storm_times.iterrows()

    def _subset_data(self, X, row):
        # Subset X by one storm

        storm_num, storm = row

        if isinstance(X, pd.Series):
            X = X.to_frame()

        start, end = storm["start_time"], storm["end_time"]
        X_ = X[start:end]
        X_["storm"] = storm_num

        return X_

    def _storm_label(self, X, row):
        # There's probably a more memory efficient way to do this
        return self._subset_data(X, row)["storm"]

    def _threshold_storms(self, y, n_storms, threshold, threshold_less_than):

        y_groupby = y.groupby(level="storm")

        if threshold_less_than:
            logger.debug("Test storms will be chosen such that min y < %s", threshold)
            min_y = y_groupby.min()
            mask = min_y < threshold
            storms_threshold = min_y[mask.values].index
        else:
            logger.debug("Test storms will be chosen such that max y > %s", threshold)
            max_y = y_groupby.max()
            mask = max_y > threshold
            storms_threshold = max_y[mask.values].index

        n_threshold = len(storms_threshold)

        # XXX
        # If # of storms that meet threshold is less than number of test storms
        # to sample, then set number of test storms to # of threshold storms
        if n_threshold < n_storms:
            n_storms = n_threshold
            logger.debug(
                "n_storms is greater than number of thresholded storms. Setting n_storms to be %s",
                n_storms,
            )

        # Randomly sample thresholded storms
        test_storms = self.rng.choice(storms_threshold, size=n_storms, replace=False)

        return test_storms

    def _storms_subsetted(self, x):
        # Check if data has been subsetted for storms
        subsetted = False

        if isinstance(x.index, pd.MultiIndex):
            # Has MultiIndex

            if "storm" in x.index.names and "times" in x.index.names:
                # MultiIndex has level 'storm' and "times"
                subsetted = True

        return subsetted

    def get_storm_labels(self, x):

        if self._storms_subsetted(x):
            return x.index.to_frame().set_index(["times"])
        else:
            storm_labels_iter = (
                self._storm_label(x, row) for row in self._storm_iter()
            )
            return pd.concat(storm_labels_iter)

    def train_test_split(
        self,
        X,
        y,
        test_size=0.2,
        train_storms=None,
        test_storms=None,
        delete_storms=None,
        threshold=None,
        threshold_less_than=True,
    ):

        # If data doesn't have storm index yet, subset it using subset_data
        if not self._storms_subsetted(X):
            X = self.subset_data(X)
        if not self._storms_subsetted(y):
            y = self.subset_data(y)

        if "test" in self.storm_times:
            test_storms = self.storm_times.query("test == True").index
        elif test_storms is None:

            is_float = isinstance(test_size, float)
            is_int = isinstance(test_size, int)
            assert is_float or is_int

            if is_int and test_size >= 1:
                n_test = test_size
            elif test_size < 1:
                n_test = int(round(test_size * self.n_storms))

            if threshold is None:
                # Choose test storms randomly
                test_storms = self.rng.choice(self.storms, size=n_test)
            else:
                # Choose test storms that cross threshold
                test_storms = self._threshold_storms(
                    y, n_test, threshold, threshold_less_than
                )

        if "train" in self.storm_times:
            train_storms = self.storm_times.query("test == False").index
        if train_storms is None or len(train_storms) == 0:
            train_mask = ~self.storms.isin(test_storms)
            train_storms = self.storms[train_mask]

        if delete_storms is not None:
            test_storms = np.setdiff1d(test_storms, delete_storms)
            train_storms = np.setdiff1d(train_storms, delete_storms)

        logger.debug("There are %s test storms.", len(test_storms))
        logger.debug("There are %s train storms.", len(train_storms))

        X_train = X.reindex(train_storms, level="storm")
        y_train = y.reindex(train_storms, level="storm")
        X_test = X.reindex(test_storms, level="storm")
        y_test = y.reindex(test_storms, level="storm")

        return TRAIN(X_train, y_train), TEST(X_test, y_test)

    def subset_data(self, X):

        X_storms_iter = (self._subset_data(X, row) for row in self._storm_iter())

        X_storms = pd.concat(X_storms_iter)
        X_storms.set_index([X_storms["storm"], X_storms.index], inplace=True)
        X_storms.drop(columns=["storm"], inplace=True)

        return X_storms


def split_data_storms(
    X,
    y,
    test_size=0.2,
    storm_times="data/stormtimes.csv",
    train_storms=None,
    test_storms=None,
    delete_storms=None,
    threshold=None,
    threshold_less_than=True,
    seed=None,
    **kwargs,
):
    # Wrapper around StormSplitter.train_test_split

    storm_times = to_absolute_path(storm_times)

    splitter = StormSplitter(storm_times, seed=seed)
    groups = splitter.get_storm_labels(y)
    train, test = splitter.train_test_split(
        X,
        y,
        test_size=test_size,
        train_storms=train_storms,
        test_storms=test_storms,
        delete_storms=delete_storms,
        threshold=threshold,
        threshold_less_than=threshold_less_than,
    )

    return train, test, groups


def split_data_ts(X, y, test_size=0.2, seed=None, **kwargs):
    # NOTE: **kwargs is used to absorb unused keyword arguments from Hydra

    test_start_idx = round(y.shape[0] * (1 - test_size))
    test_start = y.index[test_start_idx]

    def _split(x):
        if is_pandas(x):
            x_train, x_test = x.loc[:test_start], x.loc[test_start:]

            # HACK: Remove overlaps if there are any
            overlap = x_train.index.intersection(x_test.index)
            x_test.drop(index=overlap, inplace=True)
        else:
            raise TypeError("x must be a pandas DataFrame or series.")

        # FIXME: Data is split before processed so it looks like there is time
        # overlap if times are resampled

        return x_train, x_test

    X_train, X_test = _split(X)
    y_train, y_test = _split(y)

    # Groups is none
    return TRAIN(X_train, y_train), TEST(X_test, y_test), None


# Wrapper for the available split methods
def split_data(method, X, y, test_size=0.2, seed=None, **kwargs):

    logger.debug(f"Splitting data by method: {method}")

    # if method == "storms":
    if re.match("storms_.", method):
        return split_data_storms(X, y, test_size=test_size, seed=seed, **kwargs)
    elif method == "timeseries":
        return split_data_ts(X, y, test_size=test_size, seed=seed, **kwargs)
    else:
        raise ValueError(f"Split method {method} is not supported.")


if __name__ == "__main__":
    # Test

    from src.preprocessing.loading import load_solar_wind, load_symh

    X = load_solar_wind(end="2012-12-31")
    y = load_symh(end="2012-12-31")
    splitter = StormSplitter("data/stormtimes.csv")

    storm_labels = splitter.get_storm_labels(y)

    threshold = -100
    train, test = splitter.train_test_split(X, y, test_size=5, threshold=threshold)
    assert np.all(test.y.groupby(level="storm").min() < threshold)

    print("Passed!")
