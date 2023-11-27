import numpy as np
import pandas as pd
import scipy.signal as signal
import json
from .config import *


def update_config(**kwargs):
    for key, value in kwargs.items():
        if key in globals():
            globals()[key] = value
        else:
            raise ValueError(f"Unknown configuration variable: {key}")


def to_numeric_list(lst):
    try:
        return [float(item) for item in lst]
    except ValueError:
        return None


def make_label(entry):
    try:
        content = json.loads(entry)
        labels = [f"{item['type']}: {item['summary']['volume']}" for item in content]
        return ', '.join(labels)
    except Exception as e:
        return str(e)


def has_invalid_nan_count(trace):
    if trace is None:
        return True

    max_nan_count = round(MAX_NAN_COUNT * len(trace))
    nan_count = np.sum(np.isnan(trace))

    return nan_count > max_nan_count


def has_extreme_outliers(trace):
    if trace is None:
        return False

    for element in trace:
        is_outlier = element < PPI_Y_MIN_EXTREME_OUTLIER
        if is_outlier:
            return True

    return False


def is_short_trace(trace):
    if trace is None:
        return True

    if len(trace) < SHORT_TRACE_LENGTH:
        return True

    return False


def interpolate_trace(trace):
    if trace is None:
        return None
    return pd.Series(trace).interpolate().values


def filter_trace(trace):
    if trace is None:
        return None
    b, a = signal.butter(2, 0.5, 'low')
    trace = signal.filtfilt(b, a, trace)
    return trace


def normalize_trace(trace, min_val, max_percentile):
    if trace is None:
        return None
    else:
        range_val = max_percentile - min_val
        return [(x - min_val) / range_val if range_val != 0 else 0 for x in trace]


def baseline_correct_trace(trace):
    if trace is None:
        return None
    return trace - np.mean(trace[:BASELINE_LENGTH])


def calculate_threshold(group, percentile):
    return np.nanpercentile(group, percentile)


def calculate_percentiles(group, column_name):
    max_percentile = np.nanpercentile(
        group[column_name],
        PERCENTILE_THRESHOLD
    )
    return pd.Series({'max_percentile': max_percentile})


def get_max(trace, begin, end):
    if trace is None:
        return None

    start_window = int(begin / MS_PER_FRAME)
    end_window = int(end / MS_PER_FRAME)
    max_val = np.max(trace[start_window:end_window])

    return max_val
