import numpy as np
import pandas as pd
import warnings

from sklearn.tree import DecisionTreeClassifier as tree
from sklearn.linear_model import RidgeClassifier as rc

from vgc_clf.sampler import Sampler
from vgc_clf.ensemble import Ensemble
from vgc_clf.utils import data_frame_utils as df_utils

warnings.filterwarnings(action="ignore")

df = pd.read_csv("/home/imanol/Baixades/vgc_data_kids.txt", sep=",")

df.loc[np.isnan(df.sex), "sex"] = -1
for col in df.columns:
    if df[col].isna().sum() > 0:
        df.loc[df[col].isna(), col] = df[~df[col].isna()][col].mean()

healthy_cond = (df.diagnosis == 0) & (df.category == "School")
control_cond = (df.diagnosis == 0) & (df.category != "School")
adhd_cond = df.diagnosis == 1
df["group"] = ["ADHD"] * len(df)
df.loc[healthy_cond, "group"] = "healthy_control"
df.loc[control_cond, "group"] = "clinical_control"

id_variables = ["patient", "group"]
trial_encoders = ["condition", "side", "target", "correct_response"]
signal_variables = [col for col in df.columns if "feat_" in col][210:]
side_variables = ["trial_num", "sti_time", "reaction_time", "red_frog_time", "broken_fixations",
                  "total_session_time", "age", "sex", "oculomotor_deficiences", "school_performance"]
target_variable = "diagnosis"

df_dgn = df_utils.get_patients_data_frame(df=df, patient_column_name="patient",
                                          patient_info_columns=["patient", "group", "age", "sex",
                                                                "oculomotor_deficiences", "school_performance",
                                                                "diagnosis"])

cv_batches = 10
cv_dfs = df_utils.generate_cross_validation_batch(n_batches=cv_batches, signal_df=df, patients_df=df_dgn,
                                                  patient_id_column="patient",
                                                  strata="group",
                                                  strata_delimiter="ADHD", test_size=10)

for k, (df_fit, df_val, train_index, _) in enumerate(cv_dfs):

    print(f"------ITERATION {k + 1} of {cv_batches}", flush=True)

    patients_fit = list(df_dgn.loc[train_index, "patient"])

    train_patients = np.random.choice(patients_fit, size=int(np.ceil(0.8 * len(patients_fit))), replace=False)
    test_patients = list(set(patients_fit) - set(train_patients))

    df_train = df_fit[df_fit.patient.isin(train_patients)].copy()
    df_test = df_fit[df_fit.patient.isin(test_patients)].copy()

    train_samplings = Sampler()
    test_samplings = Sampler()

    valid_variables = trial_encoders + signal_variables + side_variables

    train_samplings.generate_batches(df_train,
                                     n_batches=100,
                                     batch_len=1600,
                                     target_variable=target_variable,
                                     input_variables=valid_variables,
                                     verbose=True)

    test_samplings.generate_batches(df_test,
                                    n_batches=100,
                                    batch_len=400,
                                    target_variable=target_variable,
                                    input_variables=valid_variables,
                                    verbose=True)

    classifier_list = [rc, tree]
    node_sizes = [750, 250]
    kwargs_list = [{"alpha": 1.2},
                   {"max_depth": 30, "max_features": 30, "min_samples_leaf": 30}]

    vgc_classifier = Ensemble(classifier_list=classifier_list, node_sizes=node_sizes, kwargs_list=kwargs_list)
    vgc_classifier.fit(batch_list_train=train_samplings, batch_list_test=test_samplings,
                       score_cap=0.6, get_best=100, verbose=True)

    prd = vgc_classifier.predict(df=df_val, verbose=True)

    print(f"---SCORE: {(prd == df_val[target_variable]).mean()}", flush=True)
