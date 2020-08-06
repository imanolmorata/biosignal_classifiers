import pandas as pd

from b2s_clf.experiments.experiment import Experiment
from b2s_clf.utils import data_frame_utils as df_utils


class LeaveOneOutExperiment(Experiment):
    """
    A class that calls parent class Experiment to run a leave-one-out (LOO) cross-validation experiment.The class works
    by reading a bunch of information stored in different dictionaries. Refer to /json_examples and library
    b2s_clf/apps/ for specific information and templates on these dictionaries.
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
        self.experiment_type = "LEAVE-ONE-OUT VALIDATION"
        self.performance_method = "_compute_loo_performance"
        self.dataframe_method = "_get_loo_data_frame"

    def run(self, loo_variable, verbose=False):
        """
        Calls b2s_clf.experiments.Experiment._run() with specific LOO cross_validation parameters.

        Args:
            loo_variable: Strata or discrete variable that admits a LOO routine.
            verbose: Whether to print progress on screen.

        """
        self.cv_dfs = df_utils.generate_leave_one_out_batch(signal_df=self.df, subjects_df=self.subject_df,
                                                            subject_id_column=self.subject_column,
                                                            strata_column=loo_variable)

        cv_batches = len(self.df[loo_variable].unique())
        self._run(cv_batches=cv_batches, verbose=verbose)

    def _compute_loo_performance(self, prd, prd_prb, df_val, df_fit):
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
                       prd_prb]

        return performance

    @staticmethod
    def _get_loo_data_frame(scores):
        """
        Transforms performance scores into a data frame.

        Args:
            scores: List with performance metrics for each iteration of the experiment.

        Returns:
            pandas.DataFrame with structured performance data.

        """
        return pd.DataFrame(scores, columns=["accuracy", "FNR", "FPR", "positive class probability"])
