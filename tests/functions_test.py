import unittest
import numpy as np
import pandas as pd

from blinklab_python_sdk.functions import update_config, to_numeric_list, make_label, has_invalid_nan_count, \
    has_extreme_outliers, is_short_trace, interpolate_trace, filter_trace, normalize_trace, baseline_correct_trace, \
    calculate_threshold, calculate_percentiles, get_max


class TestFunctions(unittest.TestCase):

    def setUp(self):
        self.trace = np.array([1, 2, 3, 4, 5, np.nan, 7, 8, 9, 10])
        self.json_entry = '[{"type": "type1", "summary": {"volume": "volume1"}}, {"type": "type2", "summary": {"volume": "volume2"}}]'

    def test_update_config_with_unknown_variable(self):
        with self.assertRaises(ValueError):
            update_config(UNKNOWN_VARIABLE=10)

    def test_to_numeric_list_with_valid_input(self):
        result = to_numeric_list(['1', '2', '3'])
        self.assertEqual(result, [1.0, 2.0, 3.0])

    def test_to_numeric_list_with_invalid_input(self):
        result = to_numeric_list(['1', 'two', '3'])
        self.assertIsNone(result)

    def test_make_label_with_valid_input(self):
        result = make_label(self.json_entry)
        self.assertEqual(result, 'type1: volume1, type2: volume2')

    def test_has_invalid_nan_count_with_valid_input(self):
        result = has_invalid_nan_count(self.trace)
        self.assertFalse(result)

    def test_has_invalid_nan_count_with_invalid_input(self):
        result = has_invalid_nan_count(None)
        self.assertTrue(result)

    def test_has_extreme_outliers_with_valid_input(self):
        result = has_extreme_outliers(self.trace)
        self.assertFalse(result)

    def test_has_extreme_outliers_with_invalid_input(self):
        result = has_extreme_outliers(None)
        self.assertFalse(result)

    def test_is_short_trace_with_invalid_input(self):
        result = is_short_trace(None)
        self.assertTrue(result)

    def test_interpolate_trace_with_valid_input(self):
        result = interpolate_trace(self.trace)
        self.assertEqual(result.tolist(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

    def test_interpolate_trace_with_invalid_input(self):
        result = interpolate_trace(None)
        self.assertIsNone(result)

    def test_filter_trace_with_valid_input(self):
        result = filter_trace(self.trace)
        self.assertIsNotNone(result)

    def test_filter_trace_with_invalid_input(self):
        result = filter_trace(None)
        self.assertIsNone(result)

    def test_normalize_trace_with_valid_input(self):
        result = normalize_trace(self.trace, 1, 10)
        self.assertIsNotNone(result)

    def test_normalize_trace_with_invalid_input(self):
        result = normalize_trace(None, 1, 10)
        self.assertIsNone(result)

    def test_baseline_correct_trace_with_valid_input(self):
        result = baseline_correct_trace(self.trace)
        self.assertIsNotNone(result)

    def test_baseline_correct_trace_with_invalid_input(self):
        result = baseline_correct_trace(None)
        self.assertIsNone(result)

    def test_calculate_percentiles_with_valid_input(self):
        result = calculate_percentiles(pd.DataFrame({'column1': self.trace}), 'column1')
        self.assertIsNotNone(result)

    # def test_get_max_with_valid_input(self):
    #     result = get_max(self.trace, 0, 1000)
    #     self.assertEqual(result, 10.0)

    def test_get_max_with_invalid_input(self):
        result = get_max(None, 0, 1000)
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
