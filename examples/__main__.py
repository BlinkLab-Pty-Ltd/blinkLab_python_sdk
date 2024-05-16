import pandas as pd
import sys
import warnings
from blinklab_python_sdk import functions as sdk

warnings.filterwarnings("ignore")

FRAME_RATE = 60
MS_PER_FRAME = 1000 / FRAME_RATE
NR_BASELINE_SAMPLES = round(500 / MS_PER_FRAME)


def preprocess(data: pd.DataFrame):
    df = data[['proto_trial_content', 'proto_trial_hash', 'eye_lid_trace_leftEyeTrace', 'trial_sortOrder',
               'session_trials_id']].copy()
    df.sort_values(by=['trial_sortOrder'], inplace=True)

    print()
    print("Number of rows:", len(df), ", Baseline length:", BASELINE_LENGTH, ", samples:", "(", NR_BASELINE_SAMPLES,
          ")")
    print()
    df['label'] = df['proto_trial_content'].apply(sdk.make_label)
    df['eye_trace'] = df['eye_lid_trace_leftEyeTrace'].apply(sdk.split_csv_trace)
    df['eye_trace'] = df['eye_trace'].apply(sdk.to_numeric_list)
    df['eye_trace'] = df['eye_trace'].apply(lambda x: [-1 * item for item in x] if x is not None else None)

    # Remove rows with no eye_trace
    df = df[df['eye_trace'].notnull()]

    # make baseline 500 ms
    if BASELINE_LENGTH > 500:
        print(f"Resampling baseline length from {BASELINE_LENGTH} to 500 ms...")
        df['eye_trace'] = df.apply(
            lambda row: sdk.resample_trace(
                row['eye_trace'],
                BASELINE_LENGTH,
                500),
            axis=1)
        print(f"Resampling done.")
        print()

    #####
    # Handle system latency
    #####
    if SYSTEM_LATENCY > 0:
        print(f"Handling system latency of {SYSTEM_LATENCY}...")
        system_latency_frames = round(SYSTEM_LATENCY / MS_PER_FRAME)
        print(f"System latency in frames: {system_latency_frames}")
        df['eye_trace'] = df['eye_trace'].apply(lambda x: x[system_latency_frames:] if x is not None else None)

    df['baseline_frames'] = df['proto_trial_content'].apply(sdk.get_baseline_frames) + NR_BASELINE_SAMPLES

    #####
    # Cleanse NaN values
    #####
    df['eye_trace'] = df['eye_trace'].apply(sdk.cleanse_nan_trace)
    df['nan_percentage'] = df['eye_trace'].apply(sdk.calculate_nan_percentage)
    df['eye_trace'] = df.apply(lambda row: None if row['nan_percentage'] > 25 else row['eye_trace'], axis=1)

    # #####
    # # make traces same length, per proto_trial_hash take the min length and truncate the rest
    # #####
    # df['trace_length'] = df['eye_trace'].apply(lambda x: len(x) if x is not None else None)
    # df['min_trace_length'] = df.groupby('proto_trial_hash')['trace_length'].transform('min').astype(int)
    #
    # print(df[['proto_trial_hash', 'min_trace_length']].drop_duplicates())
    #
    # df['eye_trace'] = df.apply(
    #     lambda row: row['eye_trace'][:row['min_trace_length']] if row['eye_trace'] is not None else None, axis=1)
    # df['max_trace_length'] = df.groupby('proto_trial_hash')['trace_length'].transform('max').astype(int)
    #
    # print(df[['proto_trial_hash', 'max_trace_length']].drop_duplicates())

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
            row['filtered_trace'],
            row['trial_sortOrder'],
            row['baseline_frames']
        ), axis=1)

    df['first_baseline_corrected_trace'] = df['first_baseline_corrected_trace'].apply(
        lambda x: x.tolist() if x is not None else None)

    ######
    # First Normalization
    ######
    all_traces = df['first_baseline_corrected_trace'].dropna().tolist()
    global_min, global_max = sdk.calculate_global_percentile_min_max(all_traces)

    df['first_normalized_trace'] = df['first_baseline_corrected_trace'].apply(
        lambda trace: sdk.percentile_min_max_scale_trace(trace, global_max, global_min) if trace is not None else None)

    threshold = sdk.get_outlier_threshold(df, 'first_normalized_trace', NR_BASELINE_SAMPLES, -0.4)

    ##### Remove traces below threshold
    df['normalized_trace_removed_outliers'] = df.apply(
        lambda row: sdk.remove_traces_below_threshold(
            row['first_normalized_trace'],
            row['trial_sortOrder'],
            threshold
        ),
        axis=1)
    print()
    print("First normalization done.")
    print()

    ######
    # Second baseline correction
    ######
    df['second_baseline_corrected_trace'] = df.apply(
        lambda row: sdk.baseline_correct_trace(
            row['normalized_trace_removed_outliers'],
            row['trial_sortOrder'],
            row['baseline_frames']),
        axis=1)

    df['second_baseline_corrected_trace'] = df.apply(
        lambda row: sdk.remove_baseline_blinks(
            row['second_baseline_corrected_trace'],
            row['baseline_frames'],
            row['trial_sortOrder']
        ),
        axis=1)

    threshold = sdk.get_outlier_threshold(df, 'second_baseline_corrected_trace', NR_BASELINE_SAMPLES, -0.4)

    ##### Remove traces below threshold
    df['second_baseline_corrected_trace_removed_outliers'] = df.apply(
        lambda row: sdk.remove_traces_below_threshold(
            row['second_baseline_corrected_trace'],
            row['trial_sortOrder'],
            threshold
        ),
        axis=1)

    df['second_baseline_corrected_trace_removed_outliers'] = df[
        'second_baseline_corrected_trace_removed_outliers'].apply(
        lambda x: x.tolist() if x is not None else None)

    ######
    # Second Normalization
    ######
    all_traces = df['second_baseline_corrected_trace_removed_outliers'].dropna().tolist()
    global_min, global_max = sdk.calculate_global_percentile_min_max(all_traces)

    df['second_normalized_trace'] = df['second_baseline_corrected_trace_removed_outliers'].apply(
        lambda trace: sdk.percentile_min_max_scale_trace(trace, global_max) if trace is not None else None)

    print("Processing done.")

    df.rename(columns={
        'normalized_trace_removed_outliers': 'normalized_trace',
        'second_normalized_trace': 'baseline_trace',

    }, inplace=True)

    return df


if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    BASELINE_LENGTH = int(sys.argv[3])
    SYSTEM_LATENCY = int(sys.argv[4])
    full_df = pd.read_csv(input_file)
    processed_df = preprocess(full_df)
    processed_df.to_csv(output_file, index=False)
