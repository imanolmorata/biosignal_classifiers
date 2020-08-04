import numpy as np
import pandas as pd

from vgc_clf.dataset_transformer.signal_compressor import SignalCompressor
from vgc_clf.utils import data_frame_utils as df_utils

df = pd.read_csv("/home/imanol/Escriptori/vgc_data_kids_clean.csv", sep=",")
subjects_df = df_utils.get_subjects_data_frame(df, subject_column_name="patient",
                                               subject_info_columns=["patient", "group", "population", "age", "sex",
                                                                     "oculomotor_deficiences",
                                                                     "school_performance", "diagnosis"])
df_train, df_test, _, _ = df_utils.get_train_validation_from_data_frame(signal_df=df,
                                                                        subjects_df=subjects_df,
                                                                        subject_id_column="patient",
                                                                        target_variable="diagnosis",
                                                                        test_size=20)

n_clusters_list = [4, 5, 7]
input_cols_list = [[col for col in df_train.columns if "feat_" in col][:105],
                   [col for col in df_train.columns if "feat_" in col][105:210],
                   [col for col in df_train.columns if "feat_" in col][210:]]
apply_estimator_list = [np.mean, np.mean, np.median]

compressor_obj = SignalCompressor(n_clusters_list=n_clusters_list,
                                  input_cols_list=input_cols_list,
                                  apply_estimator_list=apply_estimator_list)

compressor_obj.fit(df=df_train, verbose=True)
df_train = compressor_obj.transform(df=df_train, verbose=True)
df_test = compressor_obj.transform(df=df_test, verbose=True)

assert len(compressor_obj.compressed_signal_column_names["compression_1"]["compression_names"]) == 4
assert len(compressor_obj.compressed_signal_column_names["compression_2"]["compression_names"]) == 5
assert len(compressor_obj.compressed_signal_column_names["compression_3"]["compression_names"]) == 7
