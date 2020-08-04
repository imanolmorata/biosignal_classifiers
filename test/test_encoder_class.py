import pandas as pd

from vgc_clf.dataset_transformer.encoder import Encoder
from vgc_clf.utils import data_frame_utils as df_utils
from vgc_clf.utils import transformer_utils as tf_utils

df = pd.read_csv("/home/imanol/Escriptori/vgc_data_kids_clean.csv", sep=",")
subjects_df = df_utils.get_subjects_data_frame(df, subject_column_name="patient",
                                               subject_info_columns=["patient", "group", "population", "age", "sex",
                                                                     "oculomotor_deficiences",
                                                                     "school_performance", "diagnosis"])
df_train, df_test,_, _ = df_utils.get_train_validation_from_data_frame(signal_df=df,
                                                                       subjects_df=subjects_df,
                                                                       subject_id_column="patient",
                                                                       target_variable="diagnosis",
                                                                       test_size=20)

encoder_list = [trf for trf in tf_utils.get_transformer_objects(transformer_list=["one_hot", "target_encoder"])]
kwargs_list = [{"drop_invariant": False}, {"drop_invariant": False, "min_samples_leaf": 30}]
input_cols_list = [["oculomotor_deficiences", "school_performance"], ["age", "sex"]]
target_col_list = [None, "diagnosis"]

encoder_obj = Encoder(transformer_list=encoder_list,
                      kwargs_list=kwargs_list,
                      input_cols_list=input_cols_list,
                      target_col_list=target_col_list)

encoder_obj.fit(df=df_train, verbose=True)
df_train = encoder_obj.fit_transform(df=df_train, verbose=True)
df_test = encoder_obj.transform(df=df_test, verbose=True)

assert "school_performance" not in df_train.columns and "school_performance" not in df_test.columns
assert "oculomotor_deficiences" not in df_train.columns and "oculomotor_deficiences" not in df_test.columns

assert "school_performance_1" in df_train.columns and "school_performance_1" in df_test.columns
assert "oculomotor_deficiences_1" in df_train.columns and "oculomotor_deficiences_1" in df_test.columns

assert any([0 < x < 1 for x in df_train["age"]])
assert any([0 < x < 1 for x in df_train["sex"]])
assert any([0 < x < 1 for x in df_test["age"]])
assert any([0 < x < 1 for x in df_test["sex"]])
