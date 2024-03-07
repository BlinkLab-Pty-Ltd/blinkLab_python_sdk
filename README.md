## BlinkLab Python SDK

This SDK provides a collection of functions designed for data analysis, particularly focusing on trace analysis and
signal processing. It offers capabilities such as converting lists to numeric formats, analyzing and normalizing traces,
and calculating various metrics.

### Installation

To install the SDK, run the following command:

```bash
pip3 install git+https://github.com/BlinkLab-Pty-Ltd/blinkLab_python_sdk.git
```

### Usage

To use this SDK, import the package and create a client:

```python
from blinklab_python_sdk import functions as sdk
```

### Documentation

`to_numeric_list(lst)`
Converts a list of strings to a list of floats. Returns None if conversion fails.

`make_label(entry)`
Generates a label from a JSON entry. The function expects a specific format for the JSON.

`has_invalid_nan_count(trace)`
Checks if a trace has more NaN values than the allowable threshold. The threshold is configurable.

`has_extreme_outliers(trace)`
Determines if a trace contains values that are considered extreme outliers.

`is_short_trace(trace)`
Evaluates if a trace is shorter than a defined length.

`interpolate_trace(trace)`
Interpolates a given trace to fill in missing data points.

`filter_trace(trace)`
Applies a low-pass Butterworth filter to the trace.

`normalize_trace(trace, min_val, max_percentile)`
Normalizes a trace based on provided minimum value and maximum percentile.

`baseline_correct_trace(trace)`
Applies baseline correction to a trace.

`calculate_threshold(group, percentile)`
Calculates the threshold value of a group at a specified percentile.

`calculate_percentiles(group, column_name)`
Calculates percentiles for a specified column in a group.

`get_max(trace, begin, end)`
Finds the maximum value in a trace within a specified time window.

### Configuration

The SDK can be configured by setting the following environment variables:

```python
sdk.update_config(MAX_NAN_COUNT=200, PPI_Y_MIN_EXTREME_OUTLIER=-10.0)
```

Check config.py for a full list of configurable variables.