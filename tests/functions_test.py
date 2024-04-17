import unittest
import numpy as np
import pandas as pd

from blinklab_python_sdk.functions import to_numeric_list, make_label, has_invalid_nan_count, \
    interpolate_trace, filter_trace, normalize_trace, baseline_correct_trace


class TestFunctions(unittest.TestCase):

    def setUp(self):
        self.trace = np.array([1, 2, 3, 4, 5, np.nan, 7, 8, 9, 10])
        self.json_entry = '[{"type": "type1", "summary": {"volume": "volume1"}}, {"type": "type2", "summary": {"volume": "volume2"}}]'

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
        result = baseline_correct_trace(self.trace, 2, 0.2, 80)
        self.assertIsNotNone(result)

    def test_baseline_correct_trace_with_invalid_input(self):
        result = baseline_correct_trace(None, 2, 0.2, 80)
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()
