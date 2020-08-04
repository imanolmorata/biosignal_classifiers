import argparse
import json
import numpy as np
import pandas as pd
import warnings

from b2s_clf.ensemble.ensemble import Ensemble
from b2s_clf.sampler.sampler import Sampler
from b2s_clf.utils import data_frame_utils as df_utils
from b2s_clf.utils import ensemble_utils as ens_utils
from b2s_clf.utils import transformer_utils as trf_utils
from b2s_clf.utils import compressor_utils as cm_utils
from b2s_clf.utils import experiments_utils as exp_utils

warnings.filterwarnings(action="ignore")


def run_leave_one_out_experiment(df, loo_variable, subject_dictionary, sampler_dictionary, ensemble_dictionary,
                                 transformer_dictionary, verbose=False):
    subject_column = subject_dictionary["subject_id_column"]
    subject_info = subject_dictionary["subject_data_columns"]
    subject_target = subject_dictionary["target_variable"]

    fraction = sampler_dictionary["train_test_fraction"]
    use_variables = sampler_dictionary["input_variables"]
    target_variable = sampler_dictionary["target_variable"]
    n_train_batches = sampler_dictionary["n_train_batches"]
    train_batches_size = sampler_dictionary["train_batches_size"]
    n_test_batches = sampler_dictionary["n_test_batches"]
    test_batches_size = sampler_dictionary["test_batches_size"]

    classifier_list = [clf for clf in ens_utils.get_classifier_objects(ensemble_dictionary["classifier_list"])]
    node_sizes = ensemble_dictionary["node_sizes"]
    classifier_kwargs_list = ensemble_dictionary["kwargs_list"]
    score_cap = ensemble_dictionary["score_cap"]
    get_best = ensemble_dictionary["get_best"]
    class_threshold = ensemble_dictionary["class_threshold"]

    encoder_list = [enc for enc in trf_utils.get_transformer_objects(transformer_dictionary["Encoders"])]
    encoder_kwargs = transformer_dictionary["Encoders_kwargs"]
    encoders_input_columns = transformer_dictionary["Encoders_input_columns"]
    encoders_target_columns = transformer_dictionary["Encoders_target_columns"]

    normalizers_list = [nrm for nrm in trf_utils.get_transformer_objects(transformer_dictionary["Normalizers"])]
    normalizers_kwargs = transformer_dictionary["Normalizers_kwargs"]
    normalizers_input_columns = transformer_dictionary["Normalizers_input_columns"]

    signal_compressor_clusters = transformer_dictionary["Signal_compressor_clusters"]
    signal_compressor_input_columns = transformer_dictionary["Signal_compressor_input_columns"]
    signal_compressor_apply_functions = \
        [ap for ap in cm_utils.get_apply_functions(transformer_dictionary["Signal_compressor_apply_estimators"])]

    df_dgn = df_utils.get_subjects_data_frame(df=df, subject_column_name=subject_column,
                                              subject_info_columns=subject_info)

    loo_dfs = df_utils.generate_leave_one_out_batch(signal_df=df, subjects_df=df_dgn,
                                                    subject_id_column=subject_column,
                                                    strata_column=loo_variable)

    scores = []
    cv_batches = len(df[loo_variable].unique())
    for k, (df_fit, df_val, train_index, _) in enumerate(loo_dfs):
        print(f"------ITERATION {k + 1} of {cv_batches}", flush=True)
        valid_variables = use_variables.copy()

        if len(encoder_list) > 0:
            df_fit, df_val, valid_variables = exp_utils.transform_with_encoders(df, df_fit, df_val, valid_variables,
                                                                                encoder_list, encoder_kwargs,
                                                                                encoders_input_columns,
                                                                                encoders_target_columns,
                                                                                verbose=verbose)

        if len(normalizers_list) > 0:
            df_fit, df_val = exp_utils.transform_with_normalizers(df_fit, df_val, normalizers_list, normalizers_kwargs,
                                                                  normalizers_input_columns, verbose=verbose)

        if len(signal_compressor_clusters) > 0:
            df_fit, df_val, valid_variables = \
                exp_utils.transform_with_signal_compressors(df_fit, df_val, valid_variables, signal_compressor_clusters,
                                                            signal_compressor_input_columns,
                                                            signal_compressor_apply_functions, verbose=verbose)

        ts = int(np.ceil((1. - fraction) * len(train_index)))
        df_train, df_test, _, _ = df_utils.get_train_validation_from_data_frame(signal_df=df_fit,
                                                                                subjects_df=df_dgn.loc[train_index, :],
                                                                                subject_id_column=subject_column,
                                                                                target_variable=subject_target,
                                                                                test_size=ts)

        train_samplings = Sampler()
        test_samplings = Sampler()

        train_samplings.generate_batches(df_train,
                                         n_batches=n_train_batches,
                                         batch_len=train_batches_size,
                                         target_variable=target_variable,
                                         input_variables=valid_variables,
                                         verbose=verbose)

        test_samplings.generate_batches(df_test,
                                        n_batches=n_test_batches,
                                        batch_len=test_batches_size,
                                        target_variable=target_variable,
                                        input_variables=valid_variables,
                                        verbose=verbose)

        vgc_classifier = Ensemble(classifier_list=classifier_list, node_sizes=node_sizes,
                                  kwargs_list=classifier_kwargs_list)
        vgc_classifier.fit(batch_list_train=train_samplings, batch_list_test=test_samplings,
                           score_cap=score_cap, get_best=get_best, verbose=verbose)

        prd_prb = vgc_classifier.predict_proba(df=df_val, verbose=verbose)
        prd = (prd_prb > class_threshold) * 1
        it_performance = [(prd == df_val[target_variable]).mean(),
                          1 - (prd[df_val[target_variable] == 0] == [0] * len(
                              df_val[df_val[target_variable] == 0])).mean(),
                          1 - (prd[df_val[target_variable] == 1] == [1] * len(
                              df_val[df_val[target_variable] == 1])).mean()]
        scores.append(it_performance)

        print(f"---SCORE: {scores[-1][0]}", flush=True)

    print("------END OF LEAVE-ONE-OUT VALIDATION", flush=True)

    scores = pd.DataFrame(scores, columns=["accuracy", "FNR", "FPR"])

    print(f"Average score: {np.round(scores.accuracy.mean(), 6)}")
    print(f"Average FNR: {np.round(scores.FNR.mean(), 6)}")
    print(f"Average FPR: {np.round(scores.FPR.mean(), 6)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--data_frame", help="Path to data frame containing signal data to build the experiment.",
                        type=str, required=True)
    parser.add_argument("-v", "--loo_variable", help="Variable with respect to which generate the leave-one-out batch",
                        type=str, required=True)
    parser.add_argument("-p", "--subject_json", help="Path to json file containing subject/patient data frame "
                                                     "generation info.", type=str, required=True)
    parser.add_argument("-s", "--sampler_json", help="Path to json file containing sampler generation info.", type=str,
                        required=True)
    parser.add_argument("-e", "--ensemble_json", help="Path to json file containing ensemble generation info.",
                        type=str, required=True)
    parser.add_argument("-t", "--transformer_json", help="Path to json file containing transformer generation info.",
                        type=str, required=True)
    parser.add_argument("--verbose", help="Be verbose on progress on screen", default=False, action="store_true")

    args = parser.parse_args()

    df_in = pd.read_csv(args.data_frame, sep=",")
    subject_dict = json.load(open(args.subject_json, "r"))
    sampler_dict = json.load(open(args.sampler_json, "r"))
    ensemble_dict = json.load(open(args.ensemble_json, "r"))
    transformer_dict = json.load(open(args.transformer_json, "r"))

    run_leave_one_out_experiment(df=df_in,
                                 loo_variable=args.loo_variable,
                                 subject_dictionary=subject_dict,
                                 sampler_dictionary=sampler_dict,
                                 ensemble_dictionary=ensemble_dict,
                                 transformer_dictionary=transformer_dict,
                                 verbose=args.verbose)
