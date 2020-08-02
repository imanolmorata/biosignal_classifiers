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


def get_train_validation_from_data_frame(signal_df, patients_df, patient_id_column, target_variable, test_size):
    assert len(patients_df[target_variable].unique()) == 2

    condition = patients_df[target_variable] == 1
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


def generate_cross_validation_batch(n_batches, signal_df, patients_df, patient_id_column, target_variable, test_size):
    for _ in np.arange(n_batches):
        yield get_train_validation_from_data_frame(signal_df=signal_df,
                                                   patients_df=patients_df,
                                                   patient_id_column=patient_id_column,
                                                   target_variable=target_variable,
                                                   test_size=test_size)
