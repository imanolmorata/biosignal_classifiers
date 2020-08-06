"""
App to run a leave-one-out cross-validation experiment with random sampling. Paths to json files containing the
experiment  build instructions are needed. Such instructions are classified in four kinds:
* Data structure
* Sampling build
* Ensemble build
* Data set transformation steps.

The app returns accuracy, FPR and FNR.

Example call:
    python leave_one_out_experiment.py -v target -p subject.json -s sampling.json -e ensemble.json -t transform.json
    --verbose
"""

import argparse
import json
import pandas as pd
import warnings

from b2s_clf.experiments.leave_one_out import LeaveOneOutExperiment

warnings.filterwarnings(action="ignore")


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

    experiment_object = LeaveOneOutExperiment(df_in, subject_dict, sampler_dict, ensemble_dict, transformer_dict)
    experiment_object.run(loo_variable=args.loo_variable, verbose=args.verbose)
    print(experiment_object.experiment_stats)
