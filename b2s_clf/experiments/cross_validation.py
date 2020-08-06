import pandas as pd

from sklearn.metrics import roc_auc_score

from b2s_clf.experiments.experiment import Experiment
from b2s_clf.utils import data_frame_utils as df_utils


class CrossValidationExperiment(Experiment):

    def __init__(self, df, subject_dictionary, sampler_dictionary, ensemble_dictionary, transformer_dictionary):
        super().__init__(df, subject_dictionary, sampler_dictionary, ensemble_dictionary, transformer_dictionary)
        self.experiment_type = "CROSS VALIDATION"
        self.performance_method = "_compute_cv_performance"
        self.dataframe_method = "_get_cv_data_frame"

    def run(self, cv_batches, test_set_size, verbose=False):
        self.cv_dfs = df_utils.generate_cross_validation_batch(n_batches=cv_batches,
                                                               signal_df=self.df,
                                                               subjects_df=self.subject_df,
                                                               subject_id_column=self.subject_column,
                                                               target_variable=self.subject_target,
                                                               test_size=test_set_size)

        self._run(cv_batches=cv_batches, verbose=verbose)

    def _compute_cv_performance(self, prd, prd_prb, df_val):
        performance = [(prd == df_val[self.target_variable]).mean(),
                       1 - (prd[df_val[self.target_variable] == 0] == [0] * len(
                           df_val[df_val[self.target_variable] == 0])).mean(),
                       1 - (prd[df_val[self.target_variable] == 1] == [1] * len(
                           df_val[df_val[self.target_variable] == 1])).mean(),
                       roc_auc_score(df_val[self.target_variable], prd_prb)]

        return performance

    @staticmethod
    def _get_cv_data_frame(scores):
        return pd.DataFrame(scores, columns=["accuracy", "FNR", "FPR", "roc_auc"])
