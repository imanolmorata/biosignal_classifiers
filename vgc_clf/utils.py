import numpy as np
import pandas as pd


class DataFrameUtils:

    def __init__(self):
        pass

    @staticmethod
    def get_patients_data_frame(df, patient_column_name="patient", patient_info_columns=None):

        assert patient_column_name in list(df.columns), "Patient column not found in data frame."

        if patient_info_columns is None:
            patient_info_columns = list(df.columns)
        if patient_column_name not in patient_info_columns:
            patient_info_columns = [patient_column_name] + patient_info_columns

        df_patients = df[patient_info_columns].groupby(patient_column_name, as_index=False).max()

        return df_patients

    @staticmethod
    def get_train_validation_from_data_frame(signal_df, patients_df, patient_id_column, strata, strata_delimiter,
                                             test_size):

        condition = patients_df[strata] == strata_delimiter
        max_len = np.min([len(patients_df[~condition]), len(patients_df[condition])])

        if test_size > max_len:
            test_size = max_len

        test_ctrl = np.random.choice(patients_df[~condition].index, size=test_size)
        test_adhd = np.random.choice(patients_df[condition].index, size=test_size)

        test_idx = pd.Index(list(test_ctrl) + list(test_adhd))
        train_idx = pd.Index(set(patients_df.index) - set(test_idx))

        signal_val = signal_df.merge(patients_df.loc[test_idx, patient_id_column], on=patient_id_column, how="right")
        signal_fit = signal_df.merge(patients_df.loc[train_idx, patient_id_column], on=patient_id_column, how="right")

        return signal_val, signal_fit, train_idx, test_idx
