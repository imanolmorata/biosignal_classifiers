import argparse
import json
import numpy as np
import pandas as pd
import warnings

from vgc_clf.sampler import Sampler
from vgc_clf.ensemble import Ensemble
from vgc_clf.utils import data_frame_utils as df_utils

warnings.filterwarnings(action="ignore")


def run_cross_validation_experiment(df, cv_batches, patient_dictionary, sampler_dictionary, ensemble_dictionary):

    patient_column = patient_dictionary["patient_id_column"]

    df_dgn = df_utils.get_patients_data_frame(df=df, patient_column_name=patient_column,
                                              patient_info_columns=patient_dictionary["patient_data_columns"])

    cv_dfs = df_utils.generate_cross_validation_batch(n_batches=cv_batches, signal_df=df, patients_df=df_dgn,
                                                      patient_id_column=patient_column,
                                                      strata=patient_dictionary["patient_strata_columns"],
                                                      strata_delimiter=patient_dictionary["patient_strata_delimiter"],
                                                      test_size=10)

    fraction = sampler_dictionary["train_test_fraction"]
    valid_variables = sampler_dictionary["input_variables"]
    target_variable = sampler_dictionary["target_variable"]

    classifier_list = ensemble_dictionary["classifier_list"]
    node_sizes = ensemble_dictionary["node_sizes"]
    kwargs_list = ensemble_dictionary["kwargs_list"]

    for k, (df_fit, df_val, train_index, _) in enumerate(cv_dfs):

        print(f"------ITERATION {k + 1} of {cv_batches}", flush=True)

        patients_fit = list(df_dgn.loc[train_index, patient_column])

        train_patients = np.random.choice(patients_fit, size=int(np.ceil(fraction * len(patients_fit))), replace=False)
        test_patients = list(set(patients_fit) - set(train_patients))

        df_train = df_fit[df_fit.patient.isin(train_patients)].copy()
        df_test = df_fit[df_fit.patient.isin(test_patients)].copy()

        train_samplings = Sampler()
        test_samplings = Sampler()

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

        vgc_classifier = Ensemble(classifier_list=classifier_list, node_sizes=node_sizes, kwargs_list=kwargs_list)
        vgc_classifier.fit(batch_list_train=train_samplings, batch_list_test=test_samplings,
                           score_cap=0.6, get_best=100, verbose=True)

        prd = vgc_classifier.predict(df=df_val, verbose=True)

        print(f"---SCORE: {(prd == df_val[target_variable]).mean()}", flush=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--data_frame", help="Path to data frame containing signal data to build the experiment.",
                        type=str, required=True)
    parser.add_argument("-b", "--n_batches", help="Number of batches to generate during cross-validation.", type=int,
                        required=False, default=10)
    parser.add_argument("-p", "--patient_json", help="Path to json file containing patient/subject data frame "
                                                     "generation info.", type=str,
                        required=True)
    parser.add_argument("-s", "--sampler_json", help="Path to json file containing sampler generation info.", type=str,
                        required=True)
    parser.add_argument("-e", "--ensemble_json", help="Path to json file containing ensemble generation info.",
                        type=str, required=True)

    args = parser.parse_args()

    df = pd.read_csv(args.data_frame, sep=",")
    cv_batches = args.n_batches
    patient_dictionary = json.load(open(args.patient_json, "r"))
    sampler_dictionary = json.load(open(args.sampler_json, "r"))
    ensemble_dictionary = json.load(open(args.ensemble_json, "r"))

    run_cross_validation_experiment(df=df,
                                    cv_batches=cv_batches,
                                    patient_dictionary=patient_dictionary,
                                    sampler_dictionary=sampler_dictionary,
                                    ensemble_dictionary=ensemble_dictionary)
