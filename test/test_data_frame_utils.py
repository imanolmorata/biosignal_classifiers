import numpy as np
import pandas as pd

from vgc_clf.utils import data_frame_utils as df_utils

df = pd.read_csv("/home/imanol/Escriptori/vgc_data_kids_clean.csv", sep=",")

subjects_df = df_utils.get_subjects_data_frame(df=df, subject_column_name="patient",
                                               subject_info_columns=["patient", "group", "population", "age", "sex",
                                                                     "oculomotor_deficiences", "school_performance",
                                                                     "diagnosis"])

test_df = df_utils.get_balanced_sample_from_data_frame(df=subjects_df, balancing_variable="diagnosis")
assert test_df.diagnosis.mean() == 0.5
test_df = df_utils.get_balanced_sample_from_data_frame(df=subjects_df, balancing_variable="diagnosis", max_len=10)
assert test_df.diagnosis.mean() == 0.5 and len(test_df) == 20

df_leave, df_get, _, _ = df_utils.get_strata_samples_from_data_frame(signal_df=df, subjects_df=subjects_df,
                                                                     subject_id_column="patient",
                                                                     strata_column="population",
                                                                     strata_value="Mataro")
assert "Mataro" not in df_leave.population.unique()
assert len(df_get.population.unique()) == 1 and "Mataro" in df_get.population.unique()

_, _, idx_leave, idx_get = df_utils.get_strata_samples_from_data_frame(signal_df=df, subjects_df=subjects_df,
                                                                       subject_id_column="patient",
                                                                       strata_column="population",
                                                                       strata_value="Mataro",
                                                                       balanced_by="diagnosis")
assert subjects_df.loc[idx_leave, "diagnosis"].mean() == 0.5
assert subjects_df.loc[idx_get, "diagnosis"].mean() == 0.5

for df_leave, df_get, _, _ in df_utils.generate_leave_one_out_batch(signal_df=df, subjects_df=subjects_df,
                                                                    subject_id_column="patient",
                                                                    strata_column="population"):
    assert len(df_get["population"].unique()) == 1
    strata = df_get["population"].unique()[0]
    assert strata not in list(df_leave["population"].unique())
    assert np.sum([len(df_get["population"].unique()),
                   len(df_leave["population"].unique())]) == len(df["population"].unique())

for df_leave, df_get, _, _ in df_utils.generate_leave_one_out_batch(signal_df=df, subjects_df=subjects_df,
                                                                    subject_id_column="patient",
                                                                    strata_column="population",
                                                                    balanced_by="diagnosis"):
    assert len(df_get["population"].unique()) == 1
    strata = df_get["population"].unique()[0]
    assert strata not in list(df_leave["population"].unique())
    assert np.sum([len(df_get["population"].unique()),
                   len(df_leave["population"].unique())]) <= len(df["population"].unique())

for df_leave, df_get, _, _ in df_utils.generate_leave_one_out_batch(signal_df=df, subjects_df=subjects_df,
                                                                    subject_id_column="patient",
                                                                    strata_column="patient"):
    assert len(df_get["patient"].unique()) == 1
    strata = df_get["patient"].unique()[0]
    assert strata not in list(df_leave["patient"].unique())
    assert np.sum([len(df_get["patient"].unique()),
                   len(df_leave["patient"].unique())]) == len(df["patient"].unique())
