import pandas as pd
import numpy as np
import sys
import warnings
from blinklab_python_sdk import functions as sdk

warnings.filterwarnings("ignore")


def preprocess(data: pd.DataFrame):
    df = data[['proto_trial_content', 'proto_trial_hash', 'eye_lid_trace_leftEyeTrace', 'trial_sortOrder']].copy()
    df.sort_values(by=['trial_sortOrder'], inplace=True)

    print()
    print("Processing data...")
    print("Baseline length: ", BASELINE_LENGTH, " samples", " (", NR_BASELINE_SAMPLES, ")")
    print("PPI_Y_MIN_EXTREME_OUTLIER: ", PPI_Y_MIN_EXTREME_OUTLIER)
    print("Number of rows: ", len(df))
    print()

    df['label'] = df['proto_trial_content'].apply(sdk.make_label)

    df['eye_trace'] = df['eye_lid_trace_leftEyeTrace'].apply(sdk.split_csv_trace)
    df['eye_trace'] = df['eye_trace'].apply(sdk.to_numeric_list)
    df['eye_trace'] = df['eye_trace'].apply(lambda x: [-1 * item for item in x] if x is not None else None)

    #####
    # Handle system latency
    #####
    if SYSTEM_LATENCY > 0:
        print(f"Handling system latency of {SYSTEM_LATENCY}...")
        system_latency_frames = round(SYSTEM_LATENCY / 16.66666667)
        print(f"System latency in frames: {system_latency_frames}")
        df['eye_trace'] = df['eye_trace'].apply(lambda x: x[system_latency_frames:] if x is not None else None)

    #####
    # Cleanse NaN values
    #####
    df['eye_trace'] = df['eye_trace'].apply(sdk.cleanse_nan_trace)
    df['nan_percentage'] = df['eye_trace'].apply(sdk.calculate_nan_percentage)
    df['eye_trace'] = df.apply(lambda row: None if row['nan_percentage'] > 25 else row['eye_trace'], axis=1)

    #####
    # Handle extreme outliers
    #####
    df['has_extreme_outliers'] = df.apply(
        lambda row: sdk.has_extreme_outliers(row['eye_trace'], PPI_Y_MIN_EXTREME_OUTLIER), axis=1)
    df['eye_trace'] = df.apply(lambda row: None if row['has_extreme_outliers'] else row['eye_trace'], axis=1)

    ######
    # Process eye_trace
    ######
    df['interpolated_trace'] = df['eye_trace'].apply(sdk.interpolate_trace)
    df['interpolated_trace'] = df['interpolated_trace'].apply(lambda x: x.tolist() if x is not None else None)

    df['filtered_trace'] = df['interpolated_trace'].apply(sdk.filter_trace)
    df['filtered_trace'] = df['filtered_trace'].apply(lambda x: x.tolist() if x is not None else None)

    ######
    # First baseline correction
    ######
    df['first_baseline_corrected_trace'] = df.apply(
        lambda row: sdk.baseline_correct_trace(
            row['filtered_trace'], row['trial_sortOrder'], NR_BASELINE_SAMPLES
        ), axis=1)

    df['first_baseline_corrected_trace'] = df['first_baseline_corrected_trace'].apply(
        lambda x: x.tolist() if x is not None else None)

    df['first_baseline_corrected_trace'] = df.apply(
        lambda row: sdk.remove_traces_below_threshold(row['first_baseline_corrected_trace'], row['trial_sortOrder']),
        axis=1)

    ######
    # Normalization
    ######
    all_traces = df['first_baseline_corrected_trace'].dropna().tolist()
    global_min, global_max = sdk.calculate_global_percentile_min_max(all_traces)

    df['normalized_trace'] = df['first_baseline_corrected_trace'].apply(
        lambda trace: sdk.percentile_min_max_scale_trace(trace, global_min, global_max) if trace is not None else None)

    ######
    # Second baseline correction
    ######
    df['second_baseline_corrected_trace'] = df.apply(
        lambda row: sdk.baseline_correct_trace(row['normalized_trace'], row['trial_sortOrder'], NR_BASELINE_SAMPLES),
        axis=1)

    df['second_baseline_corrected_trace'] = df.apply(
        lambda row: sdk.remove_traces_below_threshold(
            row['second_baseline_corrected_trace'], row['trial_sortOrder']),
        axis=1)

    df['second_baseline_corrected_trace'] = df['second_baseline_corrected_trace'].apply(
        lambda x: x.tolist() if x is not None else None)

    print("Processing done.")

    df.rename(columns={
        'second_baseline_corrected_trace': 'baseline_trace',
    }, inplace=True)

    return df


if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    BASELINE_LENGTH = int(sys.argv[3])
    SYSTEM_LATENCY = int(sys.argv[4])
    NR_BASELINE_SAMPLES = round(BASELINE_LENGTH / 16.66666667)
    PPI_Y_MIN_EXTREME_OUTLIER = -0.43

    full_df = pd.read_csv(input_file)
    processed_df = preprocess(full_df)
    processed_df.to_csv(output_file, index=False)
