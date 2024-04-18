import numpy as np
import pandas as pd
import scipy.signal as signal
import json
import warnings

warnings.filterwarnings("ignore")

BASELINE_BLINK_THRESHOLD = 0.3
BASELINE_DETECTION_WINDOW = 3
TRACE_VALUE_LOWER_THRESHOLD = -0.12
MIN_MAX_LOWER_PERCENTILE = 1
MIN_MAX_UPPER_PERCENTILE = 99.9


def remove_baseline_blinks(trace, baseline_length, sort_order):
    """ Removes traces that contain baseline blinks"""
    if trace is None:
        return None

    trace_array = np.array(trace)
    baseline = trace_array[:baseline_length]
    window = baseline[-BASELINE_DETECTION_WINDOW:]

    for value in window:
        if value > BASELINE_BLINK_THRESHOLD:
            print(
                f"trial: {sort_order} contains baseline blinks above {BASELINE_BLINK_THRESHOLD} "
                f"({value}) (trace will be filtered out)")
            return None
    return trace


def remove_traces_below_threshold(trace, sort_order):
    """ Removes traces that contain values below the threshold"""
    if trace is None:
        return None

    trace_array = np.array(trace)

    for value in trace_array:
        if value < TRACE_VALUE_LOWER_THRESHOLD:
            print(
                f"Trace {sort_order} contains values below {TRACE_VALUE_LOWER_THRESHOLD} "
                f"({value}) (trace will be filtered out)")
            return None

    return trace


def has_extreme_outliers(trace, threshold):
    """ Checks if the trace contains extreme outliers"""
    if trace is None:
        return False

    for element in trace:
        is_outlier = element < threshold
        if is_outlier:
            return True

    return False


def cleanse_nan_trace(trace):
    """ Cleanses the trace by removing all nan traces"""
    if trace is None:
        return None

    if np.all(np.isnan(trace)):
        return None

    return trace


def calculate_global_percentile_min_max(traces, lower_percentile=MIN_MAX_LOWER_PERCENTILE,
                                        upper_percentile=MIN_MAX_UPPER_PERCENTILE):
    """ Calculates the global min and max values of the traces"""
    valid_traces = [trace for trace in traces if trace is not None]
    if not valid_traces:
        return None, None

    concatenated_traces = np.concatenate([trace for trace in valid_traces])
    global_min = np.nanpercentile(concatenated_traces, lower_percentile)
    global_max = np.nanpercentile(concatenated_traces, upper_percentile)

    return global_min, global_max


def percentile_min_max_scale_trace(trace, global_min, global_max, new_min=0, new_max=1):
    """ Scales the trace based on the global min and max values"""
    range_val = global_max - global_min

    scaled_trace = [new_min + (x - global_min) * (new_max - new_min) / range_val if range_val != 0 else new_min for x in
                    trace]

    return scaled_trace


def calculate_nan_percentage(trace):
    """ Calculates the percentage of nan values in a trace"""
    if trace is None:
        return 100

    return round(sum(np.isnan(trace)) / len(trace) * 100, 1)


def baseline_correct_trace(trace, trial_sort_order, baseline_length):
    """ Corrects the trace based on the baseline value"""
    if trace is None or np.all(np.isnan(trace)):
        print(f"baseline_correct_trace: {trial_sort_order} is None or all nans, returning None")
        return None

    trace = np.array(trace)
    trace = trace + 10

    baseline = trace[:baseline_length]
    baseline_value = round(np.nanmedian(baseline), 4)

    if np.isnan(baseline_value):
        print(f"baseline_correct_trace: {trial_sort_order} baseline_value is nan, returning None")
        return None

    return trace - baseline_value


def split_csv_trace(x):
    """ Splits a string of comma separated values into a list of strings"""
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
    """ Converts a list of strings to a list of floats"""
    if lst is None:
        return None

    if not isinstance(lst, list):
        return None

    try:
        return [float(item) for item in lst]
    except ValueError:
        return None


def normalize_trace(trace, min_val, max_percentile):
    """ Normalizes the trace using min_val and max_percentile"""
    if trace is None or min_val is None or max_percentile is None:
        return None

    range_val = max_percentile - min_val

    return [(x - min_val) / range_val if range_val != 0 else 0 for x in trace]


def make_label(entry):
    """ Extracts the labels from the proto_trial_content"""
    try:
        content = json.loads(entry)
        labels = [f"{item['type']}: {item['summary']['volume']}" for item in content]
        return ', '.join(labels)
    except Exception as e:
        return str(e)


def interpolate_trace(trace):
    """ Interpolates the trace using linear interpolation """
    if trace is None:
        return None

    return pd.Series(trace).interpolate().values


def filter_trace(trace):
    """ Filters the trace using a butterworth filter """
    if trace is None:
        return None

    b, a = signal.butter(2, 0.5, 'low')

    return signal.filtfilt(b, a, trace)
