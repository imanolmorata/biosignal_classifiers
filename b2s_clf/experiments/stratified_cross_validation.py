import pandas as pd

from sklearn.metrics import roc_auc_score

from b2s_clf.experiments.experiment import Experiment
from b2s_clf.utils import data_frame_utils as df_utils


class StratifiedCrossValidationExperiment(Experiment):
    """
    A class that calls parent class Experiment to run a stratified cross-validation (SCV) experiment.The class works by
    reading a bunch of information stored in different dictionaries. Refer to /json_examples and library b2s_clf/apps/
    for specific information and templates on these dictionaries.
    """

    def __init__(self, df, subject_dictionary, sampler_dictionary, ensemble_dictionary, transformer_dictionary):
        """
        Class constructor. Calls parent class and sets additional arguments.

        Args:
            df: pandas.DataFrame containing signal data.
            subject_dictionary: Dict with subject data build information.
            sampler_dictionary: Dict with sampling instructions.
            ensemble_dictionary: Dict with ensemble build instructions.
            transformer_dictionary: Dict with data set transformation build
        """
        super().__init__(df, subject_dictionary, sampler_dictionary, ensemble_dictionary, transformer_dictionary)
        self.experiment_type = "STRATIFIED CV"
        self.performance_method = "_compute_scv_performance"
        self.dataframe_method = "_get_scv_data_frame"

    def run(self, strata_variable, balanced_by, verbose=False):
        """
        Calls b2s_clf.experiments.Experiment._run() with specific SCV parameters.

        Args:
            strata_variable: Strata or discrete variable used to build the SCV batches.
            balanced_by: A second binary variable to balance the SCV batches upon.
            verbose: Whether to print progress on screen.

        """
        self.cv_dfs = df_utils.generate_leave_one_out_batch(signal_df=self.df, subjects_df=self.subject_df,
                                                            subject_id_column=self.subject_column,
                                                            strata_column=strata_variable,
                                                            balanced_by=balanced_by)

        cv_batches = len(self.df[strata_variable].unique())
        self._run(cv_batches=cv_batches, verbose=verbose)

    def _compute_scv_performance(self, prd, prd_prb, df_val, df_fit):
        """
        Specific way of computing experiment performance metrics.

        Args:
            prd: numpy.array with predictions.
            prd_prb: numpy.array with prediction probabilities.
            df_val: pandas.DataFrame with validation set.
            df_fit: pandas.DataFrame with training set.

        Returns:
            performance: A list with performance metrics for each iteration of the experiment.

        """
        acc_score = (prd == df_val[self.target_variable]).mean()
        if len(df_val[self.target_variable].unique()) == 1:
            roc_score = acc_score
        else:
            roc_score = roc_auc_score(df_val[self.target_variable], prd_prb)

        performance = [acc_score,
                       1 - (prd[df_val[self.target_variable] == 0] == [0] * len(
                           df_val[df_val[self.target_variable] == 0])).mean(),
                       1 - (prd[df_val[self.target_variable] == 1] == [1] * len(
                           df_val[df_val[self.target_variable] == 1])).mean(),
                       roc_score,
                       len(df_val) / (len(df_val) + len(df_fit))]

        return performance

    @staticmethod
    def _get_scv_data_frame(scores):
        """
        Transforms performance scores into a data frame.

        Args:
            scores: List with performance metrics for each iteration of the experiment.

        Returns:
            pandas.DataFrame with structured performance data.

        """
        return pd.DataFrame(scores, columns=["accuracy", "FNR", "FPR", "ROC AUC score", "strata weight"])
