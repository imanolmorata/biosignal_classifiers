import numpy as np

from b2s_clf.ensemble.ensemble import Ensemble
from b2s_clf.sampler.sampler import Sampler
from b2s_clf.utils import data_frame_utils as df_utils
from b2s_clf.utils import ensemble_utils as ens_utils
from b2s_clf.utils import transformer_utils as trf_utils
from b2s_clf.utils import compressor_utils as cm_utils
from b2s_clf.utils import experiments_utils as exp_utils


class Experiment:

    def __init__(self, df, subject_dictionary, sampler_dictionary, ensemble_dictionary, transformer_dictionary):
        self.df = df
        self._set_subject_variables(subject_dictionary)
        self._set_sampling_variables(sampler_dictionary)
        self._set_ensemble_variables(ensemble_dictionary)
        self._set_data_set_transform_variables(transformer_dictionary)

        self.cv_dfs = None
        self.experiment_type = None
        self.performance_method = None
        self.dataframe_method = None

        self.experiment_stats = None

    def _set_subject_variables(self, subject_dictionary):
        self.subject_column = subject_dictionary["subject_id_column"]
        self.subject_info = subject_dictionary["subject_data_columns"]
        self.subject_target = subject_dictionary["target_variable"]
        self.subject_df = df_utils.get_subjects_data_frame(df=self.df,
                                                           subject_column_name=self.subject_column,
                                                           subject_info_columns=self.subject_info)

    def _set_sampling_variables(self, sampler_dictionary):
        self.fraction = sampler_dictionary["train_test_fraction"]
        self.use_variables = sampler_dictionary["input_variables"]
        self.target_variable = sampler_dictionary["target_variable"]
        self.n_train_batches = sampler_dictionary["n_train_batches"]
        self.train_batches_size = sampler_dictionary["train_batches_size"]
        self.n_test_batches = sampler_dictionary["n_test_batches"]
        self.test_batches_size = sampler_dictionary["test_batches_size"]

    def _set_ensemble_variables(self, ensemble_dictionary):
        self.classifier_list = [clf for clf in ens_utils.get_classifier_objects(ensemble_dictionary["classifier_list"])]
        self.node_sizes = ensemble_dictionary["node_sizes"]
        self.classifier_kwargs_list = ensemble_dictionary["kwargs_list"]
        self.score_cap = ensemble_dictionary["score_cap"]
        self.get_best = ensemble_dictionary["get_best"]
        self.class_threshold = ensemble_dictionary["class_threshold"]

    def _set_data_set_transform_variables(self, transformer_dictionary):
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

            ts = int(np.ceil((1. - self.fraction) * len(train_index)))
            df_train, df_test, _, _ = df_utils.get_train_validation_from_data_frame(
                signal_df=df_fit,
                subjects_df=self.subject_df.loc[train_index, :],
                subject_id_column=self.subject_column,
                target_variable=self.subject_target,
                test_size=ts
            )

            train_samplings = Sampler()
            test_samplings = Sampler()

            train_samplings.generate_batches(df_train,
                                             n_batches=self.n_train_batches,
                                             batch_len=self.train_batches_size,
                                             target_variable=self.target_variable,
                                             input_variables=valid_variables,
                                             verbose=verbose)

            test_samplings.generate_batches(df_test,
                                            n_batches=self.n_test_batches,
                                            batch_len=self.test_batches_size,
                                            target_variable=self.target_variable,
                                            input_variables=valid_variables,
                                            verbose=verbose)

            classifier_object = Ensemble(classifier_list=self.classifier_list,
                                         node_sizes=self.node_sizes,
                                         kwargs_list=self.classifier_kwargs_list)
            classifier_object.fit(batch_list_train=train_samplings,
                                  batch_list_test=test_samplings,
                                  score_cap=self.score_cap,
                                  get_best=self.get_best,
                                  verbose=verbose)

            prd_prb = classifier_object.predict_proba(df=df_val, verbose=verbose)
            prd = (prd_prb > self.class_threshold) * 1
            it_performance = getattr(self.__class__, self.performance_method)(self, prd, prd_prb, df_val)
            scores.append(it_performance)

            print(f"---SCORE: {scores[-1][0]}", flush=True)

        print(f"------END OF {self.experiment_type}", flush=True)

        self.experiment_stats = getattr(self.__class__, self.dataframe_method)(scores)
