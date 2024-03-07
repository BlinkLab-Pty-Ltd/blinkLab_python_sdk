import numpy as np
import pandas as pd
import scipy.signal as signal
import json
from .config import *
import matplotlib.pyplot as plt


def build_x_trace(length: int):
    return np.arange(0, length * MS_PER_FRAME, MS_PER_FRAME)


def update_config(**kwargs):
    for key, value in kwargs.items():
        if key in globals():
            globals()[key] = value
        else:
            raise ValueError(f"Unknown configuration variable: {key}")


def split_csv_trace(x):
    if x is None or pd.isna(x):
        return None

    if not isinstance(x, str):
        return None

    split_x = x[1:-1].split(',')

    if split_x[0] == '':
        return ['0'] + split_x[1:]
    else:
        return split_x


def to_numeric_list(lst):
    if lst is None:
        return None

    if not isinstance(lst, list):
        return None

    try:
        return [float(item) for item in lst]
    except ValueError:
        return None


def calculate_percentiles(group, column_name):
    group[column_name] = group[column_name].replace([None], np.nan)

    if group[column_name].isna().all():
        max_percentile = None
    else:
        max_percentile = np.nanpercentile(
            group[column_name],
            PERCENTILE_THRESHOLD
        )
    return pd.Series({'max_percentile': max_percentile})


def normalize_trace(trace, min_val, max_percentile):
    if trace is None or min_val is None or max_percentile is None:
        return None

    range_val = max_percentile - min_val

    return [(x - min_val) / range_val if range_val != 0 else 0 for x in trace]


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

    return signal.filtfilt(b, a, trace)


def baseline_correct_trace(trace):
    if trace is None:
        return None

    return trace - np.mean(trace[:BASELINE_LENGTH])


def calculate_threshold(group, percentile):
    return np.nanpercentile(group, percentile)


def get_max(trace, begin, end):
    if trace is None:
        return None

    start_window = int(begin / MS_PER_FRAME)
    end_window = int(end / MS_PER_FRAME)
    max_val = np.max(trace[start_window:end_window])

    return max_val


def plot_median(
        data: pd.DataFrame,
        y_column: str,
        x_column: str,
        split_by: str = None,
        y_min: float = -0.1,
        y_max: float = 1,
        line: float = None
):
    plt.figure(figsize=(20, 4))
    plt.ylim(y_min, y_max)
    plt.title(y_column)

    plot_df = data.copy()
    plot_df = plot_df[~(plot_df['has_invalid_nan_count'])]
    plot_df = plot_df[~(plot_df['has_extreme_outliers'])]
    plot_df = plot_df[~(plot_df['is_short_trace'])]

    max_length = max(plot_df[y_column].apply(len))
    x_trace = build_x_trace(max_length)

    for unique_value, unique_df in plot_df.groupby(split_by):
        if not unique_df.empty:
            padded_arrays = []
            for array in unique_df[y_column]:
                padding = max_length - len(array)
                padded_array = np.pad(array, (0, padding), 'constant',
                                      constant_values=np.NaN)
                padded_arrays.append(padded_array)

            label = unique_df['label'].iloc[0]
            stacked = np.stack(padded_arrays)
            median_values_filtered_df = np.nanmedian(stacked, axis=0)

            for eye_trace in padded_arrays:
                plt.plot(x_trace, eye_trace, color='lightgray', alpha=0.2)

            plt.plot(x_trace, median_values_filtered_df, label=f'Median {label}', linewidth=2)

    if line is not None:
        plt.axhline(line, color='red', label='Line')

    plt.axvspan(1500, 1550, color='blue')
    plt.axvspan(1620, 1670, color='red')

    plt.legend()
    plt.show()
