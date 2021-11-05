#############################################################################
# storm_utils.py
#
# Description: Contains pandas accessors, functions, decorators to make dealing
# with data split by storms more convenient.
#############################################################################


import pandas as pd
import numpy as np

from functools import wraps, partial
from collections import OrderedDict
from pandas.api.extensions import (
    register_index_accessor,
    register_dataframe_accessor,
    register_series_accessor,
)

STORM_LEVEL = "storm"


def _is_index(x, multi=False):
    if multi:
        return isinstance(x, pd.MultiIndex)
    else:
        return isinstance(x, pd.Index)


def _is_pandas_data(x):
    return isinstance(x, (pd.Series, pd.DataFrame))


def _is_pandas_obj(x):
    return isinstance(x, (pd.Series, pd.DataFrame, pd.Index))


# TODO: Delete beginning _
def has_storm_index(x, allow_index=False):

    if allow_index:
        if _is_index(x):
            if STORM_LEVEL in x.names:
                return True

    if _is_pandas_data(x):
        if STORM_LEVEL in x.index.names:
            return True
    return False


@register_index_accessor("storms")
class StormIndexAccessor:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    def _validate(self, obj):
        valid = has_storm_index(obj, allow_index=True)
        if not valid:
            raise AttributeError(f"Must have '{STORM_LEVEL}' level.")

    @property
    def values(self):
        if _is_index(self._obj, multi=True):
            return self._obj.get_level_values(STORM_LEVEL)
        else:
            return self._obj

    @property
    def level(self):
        return self._obj.unique(level=STORM_LEVEL)

    @property
    def nstorms(self):
        return len(self.level)

    def has_same_storms(self, *obj):
        pass

    def get(self, storm, drop_level=False):
        # XXX: There's probably a more efficient way to do this since it's ordered
        storm_idx = self._obj.names.index(STORM_LEVEL)
        storm_filter = filter(lambda x: x[storm_idx] == storm, self._obj)
        get_idx = pd.MultiIndex.from_tuples(storm_filter, names=self._obj.names)
        if drop_level:
            return get_idx.droplevel(storm_idx)
        else:
            return get_idx

    def dict(self, ordered=True, drop_storm_index=True):
        # Returns dict where keys are storms and values are the associated indices

        assert _is_index(
            self._obj, multi=True
        ), "dict is a valid method only for MultiIndex."

        if ordered:
            d = OrderedDict(self._obj.groupby(self.values))
        else:
            d = self._obj.groupby(self.values)

        if drop_storm_index:
            for key, idx in d.items():
                d[key] = idx.droplevel(level=STORM_LEVEL)

        return d


@register_series_accessor("storms")
@register_dataframe_accessor("storms")
class StormAccessor:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        valid = has_storm_index(obj, allow_index=False)
        if not valid:
            raise AttributeError(f"Must have index with level {STORM_LEVEL}")

    @property
    def index(self):
        return self._obj.index.get_level_values(level=STORM_LEVEL)

    def get(self, storm, drop_level=True):
        return self._obj.xs(storm, level=STORM_LEVEL, drop_level=drop_level)

    @property
    def level(self):
        return self._obj.index.unique(level=STORM_LEVEL)

    @property
    def nstorms(self):
        return len(self.level)

    def dict(self, ordered=True, drop_storm_index=True):

        if ordered:
            d = OrderedDict(
                {storm: df for storm, df in self._obj.groupby(level=STORM_LEVEL)}
            )
        else:
            d = {storm: df for storm, df in self._obj.groupby(level=STORM_LEVEL)}

        if drop_storm_index:
            for storm, df in d.items():
                d[storm] = df.droplevel(level=STORM_LEVEL)

        return d

    def groupby(self, **kwargs):
        return self._obj.groupby(level=STORM_LEVEL, **kwargs)

    def resample(self, *args, **kwargs):
        return self.groupby().resample(*args, **kwargs)

    def apply(self, *args, **kwargs):
        return self.groupby().apply(*args, **kwargs)

    def has_same_storms(self, *obj):
        for x in obj:
            assert has_storm_index(x, allow_index=True)
            if not self.level.equals(x.storms.level):
                return False
        return True


def _func_storms_gen(
    func, X, y=None, *, storm_kwargs=None, drop_storms=False, **kwargs
):

    if storm_kwargs is not None:
        for arg in storm_kwargs.values():
            assert _is_pandas_obj(arg), "All storm kwargs must be pandas objects."
            assert X.storms.has_same_storms(
                arg
            ), "All storm kwargs must have same storms as X."

    if y is not None:
        assert X.storms.has_same_storms(y), "X must have same storms as y"

    for storm in X.storms.level:
        # Index storm_kwargs by storm
        if storm_kwargs is not None:
            storm_kwargs_ = {
                key: arg.storms.get(storm, drop_level=drop_storms)
                for key, arg in storm_kwargs.items()
            }
        else:
            storm_kwargs_ = {}

        if y is not None:
            # Index X, y by storm and use as argument
            yield func(
                X.storms.get(storm, drop_level=drop_storms),
                y.storms.get(storm, drop_level=drop_storms),
                **storm_kwargs_,
                **kwargs,
            )
        else:
            yield func(
                X.storms.get(storm, drop_level=drop_storms), **storm_kwargs_, **kwargs
            )


def apply_storms(
    func, X, y=None, *, storm_kwargs=None, concat="pandas", drop_storms=False, **kwargs
):

    func_storms = _func_storms_gen(
        func, X, y, storm_kwargs=storm_kwargs, drop_storms=drop_storms, **kwargs
    )

    if concat == "pandas":
        res = pd.concat(func_storms, keys=X.storms.level, copy=False)
    elif concat == "numpy":
        # XXX: There's probably a more memory efficient way to do this
        res = np.concatenate(list(func_storms))
    elif concat == "list":
        res = list(func_storms)
    elif concat is None:
        # For functions with side effects
        for _ in func_storms:
            pass
        return None

    return res


# Needed to separate this so we can writw iterate_storms for both functions and methods
def _iterate_storms_wrapper(
    func, storm_params, concat, drop_storms, X, y=None, **kwargs
):
    # Iterate through args, kwargs and see which is or has MultiIndex

    # When args, kwargs doesn't have storm index, just return result of func
    if not has_storm_index(X):
        if y is None:
            return func(X, **kwargs)
        elif not has_storm_index(y):
            return func(X, y, **kwargs)
        else:
            raise ValueError("y has storm index but X doesn't.")

    if y is not None:
        assert X.storms.has_same_storms(y), "X must have same storms as y"

    if storm_params is not None:
        storm_kwargs = {key: kwargs.pop(key) for key in storm_params}
    else:
        storm_kwargs = None

    return apply_storms(
        func,
        X,
        y,
        storm_kwargs=storm_kwargs,
        concat=concat,
        drop_storms=drop_storms,
        **kwargs,
    )


# Decorator for iterating storms
def iterate_storms_func(storm_params=None, *, concat="pandas", drop_storms=False):
    # storm_params: kwargs that have storm index
    def decorator(func):
        @wraps(func)
        def wrapper(X, y=None, **kwargs):
            return _iterate_storms_wrapper(
                func=func,
                storm_params=storm_params,
                concat=concat,
                drop_storms=drop_storms,
                X=X,
                y=y,
                **kwargs,
            )

        return wrapper

    return decorator


def iterate_storms_method(storm_params=None, *, concat="pandas", drop_storms=False):
    # storm_params: kwargs that have storm index
    def decorator(method):
        @wraps(method)
        def wrapper(inst, X, y=None, **kwargs):
            func = partial(method, inst)
            return _iterate_storms_wrapper(
                func=func,
                storm_params=storm_params,
                concat=concat,
                drop_storms=drop_storms,
                X=X,
                y=y,
                **kwargs,
            )

        return wrapper

    return decorator


# @iterate_storms()
# def test(X):
#     return X.droplevel("storm")
