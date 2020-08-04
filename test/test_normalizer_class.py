import pandas as pd

from b2s_clf.dataset_transformer.normalizer import Normalizer
from b2s_clf.utils import data_frame_utils as df_utils
from b2s_clf.utils import transformer_utils as tf_utils

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

transformer_list = [trf for trf in tf_utils.get_transformer_objects(transformer_list=["min_max_normalization",
                                                                                      "mean_var_normalization"])]
kwargs_list = [{}, {"with_mean": True, "with_std": True}]
input_cols_list = [["trial_num"], ["reaction_time", "red_frog_time"]]

normalizer_obj = Normalizer(transformer_list=transformer_list,
                            kwargs_list=kwargs_list,
                            input_cols_list=input_cols_list)

normalizer_obj.fit(df=df_train, verbose=True)
df_train = normalizer_obj.fit_transform(df=df_train, verbose=True)
df_test = normalizer_obj.transform(df=df_test, verbose=True)

assert all([0 <= x <= 1 for x in df_train["trial_num"]])
assert all([0 <= x <= 1 for x in df_test["trial_num"]])

assert any([0 < x < 1 for x in df_train["reaction_time"]])
assert any([0 < x < 1 for x in df_test["reaction_time"]])
assert any([0 < x < 1 for x in df_train["red_frog_time"]])
assert any([0 < x < 1 for x in df_test["red_frog_time"]])
