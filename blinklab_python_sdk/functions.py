import numpy as np
import pandas as pd
import scipy.signal as signal
import json
import warnings

warnings.filterwarnings("ignore")

PPI_Y_MIN_EXTREME_OUTLIER = -0.43  # cutoff for extreme outliers
MAX_NAN_COUNT = 0.25  # 25% of the trace
MS_PER_FRAME = 1 / 60 * 1000  # 16.666666666666667


def cleanse_nan_trace(trace):
    if trace is None:
        return None

    if np.all(np.isnan(trace)):
        return None

    return trace


def calculate_global_percentile_min_max(traces, lower_percentile=1, upper_percentile=99.9):
    valid_traces = [trace for trace in traces if trace is not None]
    if not valid_traces:
        return None, None

    concatenated_traces = np.concatenate([trace for trace in valid_traces])
    global_min = np.nanpercentile(concatenated_traces, lower_percentile)
    global_max = np.nanpercentile(concatenated_traces, upper_percentile)

    return global_min, global_max


def percentile_min_max_scale_trace(trace, global_min, global_max, new_min=0, new_max=1):
    range_val = global_max - global_min

    scaled_trace = [new_min + (x - global_min) * (new_max - new_min) / range_val if range_val != 0 else new_min for x in
                    trace]

    return scaled_trace


def calculate_nan_percentage(trace):
    if trace is None:
        return 100

    return round(sum(np.isnan(trace)) / len(trace) * 100, 1)


def baseline_correct_trace(trace, trial_sort_order, global_avg_baseline, baseline_length):
    print()
    print(f"baseline_correct_trace: trial: {trial_sort_order}")

    if trace is None or np.all(np.isnan(trace)):
        print("baseline_correct_trace: trace is None or all nans, returning None")
        return None

    baseline = trace[:baseline_length]
    baseline_value = round(np.nanmedian(baseline), 4)

    if np.isnan(baseline_value):
        print("baseline_correct_trace: baseline_value is nan, returning None")
        return None

    print(f"baseline_correct_trace: baseline_value {baseline_value}, global_avg_baseline {global_avg_baseline}")

    if global_avg_baseline == 0:
        print("baseline_correct_trace: global_avg_baseline is 0, returning trace")
        return trace

    percentage_diff = round(abs(baseline_value - global_avg_baseline) / abs(global_avg_baseline) * 100)
    print(f"baseline_correct_trace: percentage_diff {percentage_diff}")

    if percentage_diff > 20:
        print("baseline_correct_trace: percentage_diff > 20, correcting based on global_avg_baseline")
        correction_value = global_avg_baseline
    else:
        correction_value = baseline_value

    return trace - correction_value


def build_x_trace(length: int):
    return np.arange(0, length * MS_PER_FRAME, MS_PER_FRAME)


def split_csv_trace(x):
    if x is None or pd.isna(x):
        return None

    if not isinstance(x, str):
        return None

    split_x = x.split(',')

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


def interpolate_trace(trace):
    if trace is None:
        return None

    return pd.Series(trace).interpolate().values


def filter_trace(trace):
    if trace is None:
        return None

    b, a = signal.butter(2, 0.5, 'low')

    return signal.filtfilt(b, a, trace)
