import numpy as np
import pandas as pd


def get_patients_data_frame(df, patient_column_name="patient", patient_info_columns=None):
    assert patient_column_name in list(df.columns), "Patient column not found in data frame."

    if patient_info_columns is None:
        patient_info_columns = list(df.columns)
    if patient_column_name not in patient_info_columns:
        patient_info_columns = [patient_column_name] + patient_info_columns

    df_patients = df[patient_info_columns].groupby(patient_column_name, as_index=False).max()

    return df_patients


def get_balanced_sample_from_data_frame(df, balancing_variable, max_len=None):
    assert len(df[balancing_variable].unique()) == 2, "Passed balancing variable is not binary."

    ref_val = list(df[balancing_variable].unique())[0]

    if max_len is None:
        max_len = len(df)

    allowed_len = np.min([max_len, len(df[df[balancing_variable] == ref_val]),
                          len(df[df[balancing_variable] != ref_val])])

    condition = df[balancing_variable] == ref_val

    negative_index = np.random.choice(df[~condition].index, size=allowed_len)
    positive_index = np.random.choice(df[condition].index, size=allowed_len)

    balanced_df = pd.concat([df.loc[negative_index, :], df.loc[positive_index, :]])
    balanced_df = balanced_df.sample(frac=1.)

    return balanced_df


def get_train_validation_from_data_frame(signal_df, patients_df, patient_id_column, target_variable, test_size):
    assert len(patients_df[target_variable].unique()) == 2

    ref_val = list(patients_df[target_variable].unique())[0]

    condition = patients_df[target_variable] == ref_val
    max_len = np.min([len(patients_df[~condition]), len(patients_df[condition])])

    if test_size > max_len:
        test_size = max_len

    negative_index = np.random.choice(patients_df[~condition].index, size=test_size)
    positive_index = np.random.choice(patients_df[condition].index, size=test_size)

    index_val = pd.Index(list(negative_index) + list(positive_index))
    index_fit = pd.Index(set(patients_df.index) - set(index_val))

    signal_val = signal_df.merge(patients_df.loc[index_val, patient_id_column], on=patient_id_column, how="right")
    signal_fit = signal_df.merge(patients_df.loc[index_fit, patient_id_column], on=patient_id_column, how="right")

    return signal_fit, signal_val, index_fit, index_val


def get_strata_samples_from_data_frame(signal_df, patients_df, patient_id_column, strata_column, strata_value,
                                       balanced_by=None):
    assert strata_value in list(patients_df[strata_column].unique()), "strata_value not present in strata column."

    condition = patients_df[strata_column] == strata_value

    negative_index = patients_df[~condition].index
    positive_index = patients_df[condition].index

    if balanced_by is not None:
        check_neg = len(patients_df.loc[negative_index, balanced_by].unique()) == 2
        check_pos = len(patients_df.loc[positive_index, balanced_by].unique()) == 2

        if check_neg:
            negative_index = get_balanced_sample_from_data_frame(patients_df.loc[negative_index, :],
                                                                 balancing_variable=balanced_by).index
        if check_pos:
            positive_index = get_balanced_sample_from_data_frame(patients_df.loc[positive_index, :],
                                                                 balancing_variable=balanced_by).index

    signal_neg = signal_df.merge(patients_df.loc[negative_index, patient_id_column], on=patient_id_column, how="right")
    signal_pos = signal_df.merge(patients_df.loc[positive_index, patient_id_column], on=patient_id_column, how="right")

    return signal_neg, signal_pos, negative_index, positive_index


def generate_cross_validation_batch(n_batches, signal_df, patients_df, patient_id_column, target_variable, test_size):
    for _ in np.arange(n_batches):
        yield get_train_validation_from_data_frame(signal_df=signal_df,
                                                   patients_df=patients_df,
                                                   patient_id_column=patient_id_column,
                                                   target_variable=target_variable,
                                                   test_size=test_size)


def generate_leave_one_out_batch(signal_df, patients_df, patient_id_column, strata_column, balanced_by=None):
    for strata in patients_df[strata_column].unique():
        yield get_strata_samples_from_data_frame(signal_df=signal_df,
                                                 patients_df=patients_df,
                                                 patient_id_column=patient_id_column,
                                                 strata_column=strata_column,
                                                 strata_value=strata,
                                                 balanced_by=balanced_by)
