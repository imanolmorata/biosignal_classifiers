from b2s_clf.ensemble.ensemble_module import Ensemble
from b2s_clf.utils import data_frame_utils as df_utils
from b2s_clf.utils import ensemble_utils as ens_utils
from b2s_clf.utils import transformer_utils as trf_utils
from b2s_clf.utils import compressor_utils as cm_utils
from b2s_clf.utils import experiments_utils as exp_utils


class Experiment:
    """
    A class to perform cross-validation experiments. It supports k-fold cross-validation, leave-one-out cross-validation
    and stratified cross-validation, which implements through the corresponding sub-classes.
    """

    def __init__(self, df, subject_dictionary, sampler_dictionary, ensemble_dictionary, transformer_dictionary):
        """
        Class constructor. Will initialize all needed variables for the cross-validation experiment.

        Args:
            df: pandas.DataFrame containing signal data.
            subject_dictionary: Dict with subject data build information.
            sampler_dictionary: Dict with sampling instructions.
            ensemble_dictionary: Dict with ensemble build instructions.
            transformer_dictionary: Dict with data set transformation build
        """
        self.df = df

        self._check_dictionaries(subject_dictionary, sampler_dictionary, ensemble_dictionary, transformer_dictionary)

        self._set_subject_variables(subject_dictionary)
        self._set_sampling_variables(sampler_dictionary)
        self._set_ensemble_variables(ensemble_dictionary)
        self._set_data_set_transform_variables(transformer_dictionary)

        self.cv_dfs = None
        self.experiment_type = None
        self.performance_method = None
        self.dataframe_method = None

        self.experiment_stats = None

    @staticmethod
    def _check_dictionaries(subject_dictionary, sampler_dictionary, ensemble_dictionary, transformer_dictionary):
        """
        Checks that all experiment dictionaries contain the proper data.

        Args:
            subject_dictionary: Dict with subject data build information.
            sampler_dictionary: Dict with sampling instructions.
            ensemble_dictionary: Dict with ensemble build instructions.
            transformer_dictionary: Dict with data set transformation build

        """
        _subject_keys = ["subject_id_column", "subject_data_columns", "target_variable"]
        _sampler_keys = ["train_test_fraction", "input_variables", "target_variable", "n_train_batches",
                         "train_batches_size", "n_test_batches", "test_batches_size"]
        _ensemble_keys = ["classifier_list", "node_sizes", "kwargs_list", "score_cap", "get_best", "class_threshold"]
        _transformer_keys = ["Encoders", "Encoders_kwargs", "Encoders_input_columns", "Encoders_target_columns",
                             "Normalizers", "Normalizers_kwargs", "Normalizers_input_columns",
                             "Signal_compressor_clusters", "Signal_compressor_input_columns",
                             "Signal_compressor_apply_estimators"]

        assert all([key in subject_dictionary.keys() for key in _subject_keys]), "Missing or wrong information in " \
                                                                                 "subject dictionary"
        assert all([key in sampler_dictionary.keys() for key in _sampler_keys]), "Missing or wrong information in " \
                                                                                 "sampler dictionary"
        assert all([key in ensemble_dictionary.keys() for key in _ensemble_keys]), "Missing or wrong information in " \
                                                                                   "ensemble dictionary"
        assert all([key in transformer_dictionary.keys() for key in _transformer_keys]), "Missing or wrong " \
                                                                                         "information in transformer " \
                                                                                         "dictionary"

    def _set_subject_variables(self, subject_dictionary):
        """
        Prepares subject variables for the experiment.

        Args:
            subject_dictionary: Dict with subject data build information.

        """
        self.subject_column = subject_dictionary["subject_id_column"]
        self.subject_info = subject_dictionary["subject_data_columns"]
        self.subject_target = subject_dictionary["target_variable"]
        self.subject_df = df_utils.get_subjects_data_frame(df=self.df,
                                                           subject_column_name=self.subject_column,
                                                           subject_info_columns=self.subject_info)

    def _set_sampling_variables(self, sampler_dictionary):
        """
        Prepares sampling variables for the experiment.

        Args:
            sampler_dictionary: Dict with sampling data build information.

        """
        self.fraction = sampler_dictionary["train_test_fraction"]
        self.use_variables = sampler_dictionary["input_variables"]
        self.target_variable = sampler_dictionary["target_variable"]
        self.n_train_batches = sampler_dictionary["n_train_batches"]
        self.train_batches_size = sampler_dictionary["train_batches_size"]
        self.n_test_batches = sampler_dictionary["n_test_batches"]
        self.test_batches_size = sampler_dictionary["test_batches_size"]

    def _set_ensemble_variables(self, ensemble_dictionary):
        """
        Prepares ensemble variables for the experiment.

        Args:
            ensemble_dictionary: Dict with ensemble data build information.

        """
        self.classifier_list = [clf for clf in ens_utils.get_classifier_objects(ensemble_dictionary["classifier_list"])]
        self.node_sizes = ensemble_dictionary["node_sizes"]
        self.classifier_kwargs_list = ensemble_dictionary["kwargs_list"]
        self.score_cap = ensemble_dictionary["score_cap"]
        self.get_best = ensemble_dictionary["get_best"]
        self.class_threshold = ensemble_dictionary["class_threshold"]

    def _set_data_set_transform_variables(self, transformer_dictionary):
        """
        Prepares data set transformation variables for the experiment.

        Args:
            transformer_dictionary: Dict with data set transformation data build information.

        """
        self.encoder_list = [enc for enc in trf_utils.get_transformer_objects(transformer_dictionary["Encoders"])]
        self.encoder_kwargs = transformer_dictionary["Encoders_kwargs"]
        self.encoders_input_columns = transformer_dictionary["Encoders_input_columns"]
        self.encoders_target_columns = transformer_dictionary["Encoders_target_columns"]

        self.normalizers_list = \
            [nrm for nrm in trf_utils.get_transformer_objects(transformer_dictionary["Normalizers"])]
        self.normalizers_kwargs = transformer_dictionary["Normalizers_kwargs"]
        self.normalizers_input_columns = transformer_dictionary["Normalizers_input_columns"]

        self.signal_compressor_clusters = transformer_dictionary["Signal_compressor_clusters"]
        self.signal_compressor_input_columns = transformer_dictionary["Signal_compressor_input_columns"]
        self.signal_compressor_apply_functions = \
            [ap for ap in cm_utils.get_apply_functions(transformer_dictionary["Signal_compressor_apply_estimators"])]

    def _run(self, cv_batches, verbose=False):
        """
        Private method that runs the whole experiment with all the parameters set in previous steps. It is called from
        the sub-classes' public run method.

        Args:
            cv_batches: Number of batches in the cross-validation.
            verbose: Whether to print progress on screen.

        """
        scores = []
        for k, (df_fit, df_val, train_index, _) in enumerate(self.cv_dfs):
            print(f"------ITERATION {k + 1} of {cv_batches}", flush=True)
            valid_variables = self.use_variables.copy()

            if len(self.encoder_list) > 0:
                df_fit, df_val, valid_variables = exp_utils.transform_with_encoders(
                    self.df, df_fit, df_val,
                    valid_variables,
                    self.encoder_list,
                    self.encoder_kwargs,
                    self.encoders_input_columns,
                    self.encoders_target_columns,
                    verbose=verbose
                )

            if len(self.normalizers_list) > 0:
                df_fit, df_val = exp_utils.transform_with_normalizers(df_fit, df_val, self.normalizers_list,
                                                                      self.normalizers_kwargs,
                                                                      self.normalizers_input_columns,
                                                                      verbose=verbose)

            if len(self.signal_compressor_clusters) > 0:
                df_fit, df_val, valid_variables = \
                    exp_utils.transform_with_signal_compressors(df_fit, df_val, valid_variables,
                                                                self.signal_compressor_clusters,
                                                                self.signal_compressor_input_columns,
                                                                self.signal_compressor_apply_functions,
                                                                verbose=verbose)

            classifier_object = Ensemble(classifier_list=self.classifier_list,
                                         node_sizes=self.node_sizes,
                                         kwargs_list=self.classifier_kwargs_list)
            classifier_object.fit(df=df_fit,
                                  subject_df=self.subject_df.loc[train_index, :],
                                  subject_column=self.subject_column,
                                  target_variable=self.subject_target,
                                  split_fraction=self.fraction,
                                  fit_variables=valid_variables,
                                  n_train_batches=self.n_train_batches,
                                  n_test_batches=self.n_test_batches,
                                  train_batch_size=self.train_batches_size,
                                  test_batch_size=self.test_batches_size,
                                  score_cap=self.score_cap,
                                  get_best=self.get_best,
                                  verbose=verbose)

            prd_prb = classifier_object.predict_proba(df=df_val, verbose=verbose)
            prd = (prd_prb > self.class_threshold) * 1
            it_performance = getattr(self.__class__, self.performance_method)(self, prd, prd_prb, df_val, df_fit)
            scores.append(it_performance)

            print(f"---SCORE: {scores[-1][0]}", flush=True)

        print(f"------END OF {self.experiment_type}", flush=True)

        self.experiment_stats = getattr(self.__class__, self.dataframe_method)(scores)
