import pandas as pd

from b2s_clf.experiments.experiment import Experiment
from b2s_clf.utils import data_frame_utils as df_utils


class LeaveOneOutExperiment(Experiment):

    def __init__(self, df, subject_dictionary, sampler_dictionary, ensemble_dictionary, transformer_dictionary):
        super().__init__(df, subject_dictionary, sampler_dictionary, ensemble_dictionary, transformer_dictionary)
        self.experiment_type = "LEAVE-ONE-OUT VALIDATION"
        self.performance_method = "_compute_loo_performance"
        self.dataframe_method = "_get_loo_data_frame"

    def run(self, loo_variable, verbose=False):
        self.cv_dfs = df_utils.generate_leave_one_out_batch(signal_df=self.df, subjects_df=self.subject_df,
                                                            subject_id_column=self.subject_column,
                                                            strata_column=loo_variable)

        cv_batches = len(self.df[loo_variable].unique())
        self._run(cv_batches=cv_batches, verbose=verbose)

    def _compute_loo_performance(self, prd, prd_prb, df_val):
        performance = [(prd == df_val[self.target_variable]).mean(),
                       1 - (prd[df_val[self.target_variable] == 0] == [0] * len(
                           df_val[df_val[self.target_variable] == 0])).mean(),
                       1 - (prd[df_val[self.target_variable] == 1] == [1] * len(
                           df_val[df_val[self.target_variable] == 1])).mean()]

        return performance

    @staticmethod
    def _get_loo_data_frame(scores):
        return pd.DataFrame(scores, columns=["accuracy", "FNR", "FPR"])
