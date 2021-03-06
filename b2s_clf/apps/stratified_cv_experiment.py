"""
App to run a stratified cross-validation experiment with random sampling. Paths to json files containing the experiment
build instructions are needed. Such instructions are classified in four kinds:
* Data structure
* Sampling build
* Ensemble build
* Data set transformation steps.

The app returns accuracy, FPR, FNR and ROC AUC score.

Example call:
    python stratified_cv_experiment.py -v device -b target -p subject.json -s sampling.json -e ensemble.json
    -t transform.json --save_experiment /home/my_path --verbose
"""

import argparse
import datetime
import json
import pandas as pd
import warnings

from b2s_clf.experiments.stratified_cross_validation import StratifiedCrossValidationExperiment

warnings.filterwarnings(action="ignore")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--data_frame", help="Path to data frame containing signal data to build the experiment.",
                        type=str, required=True)
    parser.add_argument("-v", "--strata_variable", help="Variable with respect to which stratify the cross-validation"
                                                        "batch", type=str, required=True)
    parser.add_argument("-b", "--balancing_variable", help="OPTIONAL: Variable with respect to which the stratified "
                                                           "data will be balanced",
                        type=str, required=False, default=None)
    parser.add_argument("-p", "--subject_json", help="Path to json file containing subject/patient data frame "
                                                     "generation info.", type=str, required=True)
    parser.add_argument("-s", "--sampler_json", help="Path to json file containing sampler generation info.", type=str,
                        required=True)
    parser.add_argument("-e", "--ensemble_json", help="Path to json file containing ensemble generation info.",
                        type=str, required=True)
    parser.add_argument("-t", "--transformer_json", help="Path to json file containing transformer generation info.",
                        type=str, required=True)
    parser.add_argument("--save_experiment", help="Path to save experiment results.", type=str, required=False,
                        default=None)
    parser.add_argument("--verbose", help="Be verbose on progress on screen", default=False, action="store_true")

    args = parser.parse_args()

    df_in = pd.read_csv(args.data_frame, sep=",")
    subject_dict = json.load(open(args.subject_json, "r"))
    sampler_dict = json.load(open(args.sampler_json, "r"))
    ensemble_dict = json.load(open(args.ensemble_json, "r"))
    transformer_dict = json.load(open(args.transformer_json, "r"))

    experiment_object = StratifiedCrossValidationExperiment(df_in, subject_dict, sampler_dict, ensemble_dict,
                                                            transformer_dict)
    experiment_object.run(strata_variable=args.strata_variable, balanced_by=args.balancing_variable,
                          verbose=args.verbose)

    if args.save_experiment is not None:
        today_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        exp_name = f"{'_'.join(experiment_object.experiment_type.split(' '))}_{today_date}"
        experiment_object.save_experiment_results(experiment_name=exp_name, experiment_path=args.save_experiment)
    else:
        print("\nEXPERIMENT RESULTS")
        print(experiment_object.experiment_stats.mean())
