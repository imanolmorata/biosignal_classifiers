import pandas as pd

from sklearn.metrics import roc_auc_score

from b2s_clf.experiments.experiment import Experiment
from b2s_clf.utils import data_frame_utils as df_utils


class CrossValidationExperiment(Experiment):
    """
    A class that calls parent class Experiment to run a k-fold cross-validation experiment.
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
        self.experiment_type = "CROSS VALIDATION"
        self.performance_method = "_compute_cv_performance"
        self.dataframe_method = "_get_cv_data_frame"

    def run(self, cv_batches, test_set_size, verbose=False):
        """
        Calls b2s_clf.experiments.Experiment._run() with specific k-fold cross_validation parameters.

        Args:
            cv_batches: Total cross-validation batches to run.
            test_set_size: Class-wise size of the testing sample (i.e. test_size=10 --> len(test_set) = 20)
            verbose: Whether to print progress on screen.

        """
        self.cv_dfs = df_utils.generate_cross_validation_batch(n_batches=cv_batches,
                                                               signal_df=self.df,
                                                               subjects_df=self.subject_df,
                                                               subject_id_column=self.subject_column,
                                                               target_variable=self.subject_target,
                                                               test_size=test_set_size)

        self._run(cv_batches=cv_batches, verbose=verbose)

    def _compute_cv_performance(self, prd, prd_prb, df_val, df_fit):
        """
        Specific way of computing experiment performance metrics.

        Args:
            prd: numpy.array with predictions.
            prd_prb: numpy.array with prediction probabilities.
            df_val: pandas.DataFrame with validation set.
            df_fit: Unused.

        Returns:
            performance: A list with performance metrics for each iteration of the experiment.

        """
        performance = [(prd == df_val[self.target_variable]).mean(),
                       1 - (prd[df_val[self.target_variable] == 0] == [0] * len(
                           df_val[df_val[self.target_variable] == 0])).mean(),
                       1 - (prd[df_val[self.target_variable] == 1] == [1] * len(
                           df_val[df_val[self.target_variable] == 1])).mean(),
                       roc_auc_score(df_val[self.target_variable], prd_prb)]

        return performance

    @staticmethod
    def _get_cv_data_frame(scores):
        """
        Transforms performance scores into a data frame.

        Args:
            scores: List with performance metrics for each iteration of the experiment.

        Returns:
            pandas.DataFrame with structured performance data.

        """
        return pd.DataFrame(scores, columns=["accuracy", "FNR", "FPR", "ROC AUC score"])
