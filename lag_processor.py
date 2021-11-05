#############################################################################
# lag_processor.py
#
# Description: Contains class for computing lagged features
#############################################################################

import itertools
import logging
import numpy as np
import pandas as pd

from functools import partial
from sklearn.base import TransformerMixin
from pandas.tseries.frequencies import to_offset

from utils import get_freq
from storm_utils import (
    iterate_storms_method,
    apply_storms,
    StormAccessor,
    StormIndexAccessor,
    has_storm_index,
)

logger = logging.getLogger(__name__)

ZERO_MINUTES = to_offset("0T")


class LaggedFeaturesProcessor:
    """
    NOTE: X, y don't necessarily have the same freq so we can't just pass one
    combined dataframe.

    XXX: Not a proper scikit learn transformer. fit and transform take X and y.

    """

    def __init__(
        self,
        lag="0T",
        exog_lag="H",
        lead="0T",
        unit="minutes",
        history_freq=None,
        history_func="mean",
        transformer_y=None,
        njobs=1,
        return_pandas=False,
        verbose=False,
        **transformer_y_kwargs,
    ):

        self.unit = unit
        self.lag = self._process_params(lag)
        self.exog_lag = self._process_params(exog_lag)
        self.lead = self._process_params(lead)
        self.history_freq = self._process_params(history_freq)
        self.history_func = history_func

        self.njobs = njobs
        self.return_pandas = return_pandas
        self.verbose = verbose

        # NOTE: transformer_y must keep input as pd DataFrame
        # NOTE: Pass transformer that was used for X here.
        # (use PandasTransformer if required)
        self.transformer_y = transformer_y
        self.transformer_y_kwargs = transformer_y_kwargs

    def _process_params(self, param):

        if param is None:
            return None

        assert isinstance(param, (str, int))

        if isinstance(param, str):
            param = to_offset(param)
        elif isinstance(param, int):
            param = to_offset(pd.Timedelta(**{self.unit: param}))

        return param

    def _check_data(self, X, y):
        # TODO: Input validation
        # - pd Dataframe
        pass

    # TODO: Use storm accessor wherever possible
    def _compute_feature(self, target_index, X, y=None):
        """Computes ARX features to predict target at a specified time
           `target_time`.

        This method ravels subsets of `target` and `self.solar_wind` that depend on
        `self.lag` and `self.exog_lag` to obtain features to predict the target
        time series at a specified `target_time`.

        Parameters
        ----------

        target_time : datetime-like
            Time to predict.
        X : pd.DataFrame
            Exogeneous features
        y : pd.DataFrame or pd.Series
            Target time series to use as features to predict at target_time

        Returns
        -------
        np.ndarray
            Array containing features to use to predict target at target_time.

        # FIXME: Docstring outdated
        """

        # HACK: Assume time is the second element if target_index is MultiIndex tuple
        if isinstance(target_index, tuple):
            target_storm, target_time = target_index
        else:
            # TODO: Change to elif time index
            target_time = target_index

        feature = np.full(self.n_cols_, np.nan)
        end = target_time - self.lead

        # Ravel target and solar wind between start and end time
        if self.n_lag_ != 0:
            assert y is not None
            # HACK: Subset storm
            if has_storm_index(y):
                y = y.xs(target_storm, level="storm")

            start = end - self.lag
            y_ = y[start:end][::-1]

            if self.history_freq is not None:
                y_ = y_.resample(
                    self.history_freq, label="right", closed="right", origin="start"
                ).apply(self.history_func)[::-1]

            feature[: self.n_lag_] = y_[: self.n_lag_].to_numpy().ravel()

        if self.n_exog_each_col_ != 0:
            # HACK: Subset storm
            if has_storm_index(X):
                X = X.xs(target_storm, level="storm")

            start_exog = end - self.exog_lag
            X_ = X[start_exog:end][::-1][: self.n_exog_each_col_]
            feature[self.n_lag_ : self.n_cols_] = X_.to_numpy().ravel()

        # error_msg = f"Length of feature ({len(feature)}) at {target_index} != self.n_cols_ ({self.n_cols_})"
        # assert len(feature) == self.n_cols_, error_msg

        return feature

    def fit(self, X, y=None):

        # TODO: Replace with function from utils
        self.freq_X_ = get_freq(X)
        if self.lag != ZERO_MINUTES and y is not None:
            self.freq_y_ = get_freq(y)

        if self.history_freq is not None:
            # Doesn't make sense to have history_freq < y's freq since there will be
            # unnecessary NAs
            assert self.history_freq >= self.freq_y_

        if self.lag == ZERO_MINUTES:
            self.n_lag_ = 0
        else:
            if self.history_freq is not None:
                history_freq = self.history_freq
            else:
                history_freq = self.freq_y_

            self.n_lag_ = int(self.lag / history_freq)

        logger.debug("# of lagged features: %s", self.n_lag_)

        if self.exog_lag == ZERO_MINUTES:
            self.n_exog_each_col_ = 0
        else:
            self.n_exog_each_col_ = int((self.exog_lag / self.freq_X_))

        n_exog = self.n_exog_each_col_ * X.shape[1]
        logger.debug("# of exogeneous features: %s", n_exog)

        self.n_cols_ = self.n_lag_ + n_exog

        self.feature_names_ = self.get_feature_names(X, y)
        assert len(self.feature_names_) == self.n_cols_

        if self.transformer_y is not None and y is not None:
            self.transformer_y.set_params(**self.transformer_y_kwargs)
            self.transformer_y.fit(y)

        return self

    def get_feature_names(self, X, y=None):

        exog_feature_names = self._get_feature_names(
            self.exog_lag, X.columns, self.freq_X_
        )

        if y is None:
            return exog_feature_names

        if self.history_freq is None:
            lag_freq = self.freq_y_
        else:
            lag_freq = self.history_freq

        lag_feature_names = self._get_feature_names(self.lag, ["y"], lag_freq)

        # Lagged y goes first
        return lag_feature_names + exog_feature_names

    @staticmethod
    def _get_feature_names(lag, columns, freq):
        lags_timedelta = pd.timedelta_range(
            start="0 days", end=lag, freq=freq, closed="left"
        )
        # Minutes in reverse order
        lags = (int(t.total_seconds() / 60) for t in lags_timedelta)

        # Order: Iterate columns first
        # e.g. [density0, density5, ..., temperature0, temperature5, ....]
        feature_names = [f"{col}_{t}" for t, col in itertools.product(lags, columns)]

        return feature_names

    # @iterate_storms_method(drop_storms=True)
    # def _get_target(self, X, y=None):
    #     # TODO: Handle MultiIndex case

    #     if y is not None:
    #         y = y.dropna()
    #         y_start = y.index[0]
    #     else:
    #         y_start = 0

    #     max_time = max(
    #         to_offset(self.lag) + y_start, to_offset(self.exog_lag) + X.index[0]
    #     )
    #     cutoff = max_time + self.lead

    #     # FIXME
    #     return y[y.index > cutoff]

    @iterate_storms_method(drop_storms=True)
    def _get_target_index(self, X, y=None):
        # TODO: Handle MultiIndex case

        if y is not None:
            y = y.dropna()
            y_start = y.index[0]
        else:
            y_start = 0

        max_time = max(
            to_offset(self.lag) + y_start, to_offset(self.exog_lag) + X.index[0]
        )
        cutoff = max_time + self.lead

        if y is not None:
            return y.index[y.index > cutoff]
        else:
            return X.index[X.index > cutoff]

    @iterate_storms_method(["target_index"], concat="numpy", drop_storms=True)
    def _transform(self, X, y, target_index):
        # TODO: Implement parallel

        n_obs = len(target_index)

        compute_feature_ = partial(self._compute_feature, X=X, y=y)
        features_map = map(compute_feature_, target_index)

        features_iter = itertools.chain.from_iterable(features_map)
        features = np.fromiter(
            features_iter, dtype=np.float32, count=n_obs * self.n_cols_
        ).reshape(n_obs, self.n_cols_)

        return features

    def transform(self, X, y=None):
        # NOTE: Include interpolator in transformer_y if want to interpolate
        # TODO: Write tests

        if y is None:
            y_feature = None
        elif self.transformer_y is not None:
            y_feature = self.transformer_y.transform(y)
        else:
            y_feature = y

        logger.debug("Getting targets...")
        # y_target = self._get_target(X, y)
        target_index = self._get_target_index(X, y)

        logger.debug("Computing lagged features...")
        # features = self._transform(X, y_feature, target_index=y_target.index)
        features = self._transform(X, y_feature, target_index=target_index)

        assert features.shape[0] == len(target_index)

        if self.return_pandas:
            features = pd.DataFrame(
                features, index=target_index, columns=self.feature_names_
            )

        if y is None:
            return features, None
        else:
            return features, y.loc[target_index]

        # return features, y_target

    def fit_transform(self, X, y=None, **fit_params):
        # fit_transform from TransformerMixin doesn't allow y in transform
        return self.fit(X, y, **fit_params).transform(X, y)
