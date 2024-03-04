import numpy as np
import pandas as pd
import scipy.signal as signal
import json
from .config import *
import matplotlib.pyplot as plt


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

def plot_median(
        data: pd.DataFrame,
        y_column: str,
        x_column: str,
        split_by: str = None,
        y_min: float = -0.1,
        y_max: float = 1,
        line: float = None
):
    plot_df = data.copy()

    if DROP_TONES:
        plot_df = plot_df[plot_df['label'].apply(lambda x: 'tone' not in x)]

    plot_df = plot_df[~(plot_df['has_invalid_nan_count'])]
    plot_df = plot_df[~(plot_df['has_extreme_outliers'])]
    plot_df = plot_df[~(plot_df['is_short_trace'])]


    plt.figure(figsize=(20, 4))
    plt.ylim(y_min, y_max)
    plt.title(y_column)

    if split_by is None:
        for _, row in plot_df.iterrows():
            x_trace = row[x_column]
            eye_trace = row[y_column]
            plt.plot(x_trace, eye_trace, color='lightgray', alpha=0.2)

        median_values_filtered_df = np.nanmedian(np.stack(plot_df[y_column]), axis=0)
        plt.plot(plot_df[x_column].iloc[0], median_values_filtered_df, label='Median', linewidth=2)
    else:
        for unique_value, unique_df in plot_df.groupby(split_by):
            if not unique_df.empty:
                median_values_filtered_df = np.nanmedian(np.stack(unique_df[y_column]), axis=0)
                label = unique_df['label'].iloc[0]

                for _, row in unique_df.iterrows():
                    x_trace = row[x_column]
                    eye_trace = row[y_column]
                    plt.plot(x_trace, eye_trace, color='lightgray', alpha=0.2)

                if label == 'noises: 1':
                    median_pulse = np.nanmedian(np.stack(unique_df['pulse_max']), axis=0)
                    plt.axhline(median_pulse)

                plt.plot(x_trace, median_values_filtered_df, label=f'Median {label}', linewidth=2)
    if line is not None:
        plt.axhline(line, color='red', label='Line')
    plt.axvspan(1500, 1550, color='blue')
    plt.axvspan(1620, 1670, color='red')

    plt.legend()
    plt.show()