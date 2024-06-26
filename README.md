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

`interpolate_trace(trace)`
Interpolates a given trace to fill in missing data points.

`filter_trace(trace)`
Applies a low-pass Butterworth filter to the trace.

`normalize_trace(trace, min_val, max_percentile)`
Normalizes a trace based on provided minimum value and maximum percentile.

`baseline_correct_trace(trace)`
Applies baseline correction to a trace.
