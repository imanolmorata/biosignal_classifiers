import numpy as np
import pandas as pd

from b2s_clf.sampler.sampler_module import Sampler
from b2s_clf.utils import data_frame_utils as df_utils


class Ensemble:
    """
    A class that builds an ensemble model and uses it to predict new batches of data. It fits a set of weak classifiers
    on previously-sampled data and uses them to build a final voting classifier. These weak classifiers is any
    arbitrary set of sklearn-like model objects that support the methods fit, predict and predict_proba.
    """

    def __init__(self, classifier_list, node_sizes, kwargs_list):

        self.classifier_list = classifier_list
        self.node_sizes = node_sizes
        self.kwargs_list = kwargs_list
        self.nodes = []
        self.node_names = []
        self.ensemble_input_variables = None
        self.ensemble_target_variable = None

        self._check_nodes()

    def _check_nodes(self):
        """
        Checks that all weak classifiers support fit and predict methods.

        """

        assert all([hasattr(clf, "fit") for clf in self.classifier_list]), "At least one classifier object is not " \
                                                                           "compatible with 'fit' method."
        assert all([hasattr(clf, "predict") for clf in self.classifier_list]), "At least one classifier object is " \
                                                                               "not compatible with 'predict' method."

    def _check_batches(self, batch_list_train, batch_list_test):
        """
        Checks the integrity of train and test batches before fitting the ensemble.

        Args:
            batch_list_train: Object of class vgc_clf.sampler.Sampler with training batches.
            batch_list_test: Object of class vgc_clf.sampler.Sampler with test batches.

        """

        assert type(batch_list_train) == Sampler, "batch_list_train expects an object of class vgc_clf.sampler.Sampler."
        assert type(batch_list_test) == Sampler, "batch_list_test expects an object of class vgc_clf.sampler.Sampler."
        assert batch_list_train.target_variable == batch_list_test.target_variable, "Train and test have different" \
                                                                                    "target variables"
        assert all([var in batch_list_test.input_variables for var in batch_list_train.input_variables]), "Mismatch" \
                                                                                                          "in input" \
                                                                                                          "variables"
        assert all([var in batch_list_train.input_variables for var in batch_list_test.input_variables]), "Mismatch" \
                                                                                                          "in input" \
                                                                                                          "variables"

        self.ensemble_input_variables = batch_list_train.input_variables
        self.ensemble_target_variable = batch_list_train.target_variable

    @staticmethod
    def _prepare_batches(df, target_variable, subject_df, subject_column, split_fraction=0.8, fit_variables=None,
                         n_train_batches=None, n_test_batches=None, train_batch_size=None, test_batch_size=None,
                         verbose=False):
        """
        Prepares batches for the ensemble fit process.
        Args:
            df: pandas.DataFrame with signal data.
            target_variable: Target variable to fit the ensemble.
            subject_df: pandas.DataFrame with subject data.
            subject_column: subject ID columns in subject_df.
            split_fraction: Train/test sampling in batching process.
            fit_variables: Variable to consider upon fit.
            n_train_batches: Number of training batches.
            n_test_batches: Number of test batches.
            train_batch_size: Size of training batches.
            test_batch_size: Size of test batches.
            verbose: Whether to print progress on screen.

        Returns:
            dict: Train and test batches for the ensemble fit.

        """

        if fit_variables is None:
            fit_variables = list(df.columns)

        if target_variable in fit_variables:
            fit_variables.remove(target_variable)

        if n_train_batches is None:
            n_train_batches = len(subject_df[subject_df].unique())

        if n_test_batches is None:
            n_test_batches = n_train_batches

        if train_batch_size is None:
            train_batch_size = int(split_fraction * len(df))

        if test_batch_size is None:
            test_batch_size = int((1. - split_fraction) * len(df))

        ts = int(np.ceil((1. - split_fraction) * len(subject_df)))
        df_train, df_test, _, _ = df_utils.get_train_validation_from_data_frame(
            signal_df=df,
            subjects_df=subject_df,
            subject_id_column=subject_column,
            target_variable=target_variable,
            test_size=ts
        )

        train_samplings = Sampler()
        test_samplings = Sampler()

        train_samplings.generate_batches(df_train,
                                         n_batches=n_train_batches,
                                         batch_len=train_batch_size,
                                         target_variable=target_variable,
                                         input_variables=fit_variables,
                                         verbose=verbose)

        test_samplings.generate_batches(df_test,
                                        n_batches=n_test_batches,
                                        batch_len=test_batch_size,
                                        target_variable=target_variable,
                                        input_variables=fit_variables,
                                        verbose=verbose)

        return train_samplings, test_samplings

    def _filter_nodes(self, get_best):
        """
        In the event that get_best is not None when calling fit, this will filter the get_best weak classifiers and
        keep them, discarding the rest.

        Args:
            get_best: The number of best weak classifiers to keep.

        """

        clf_names = pd.DataFrame(self.node_names, columns=["idx", "classifier"])

        clf_to_choose = clf_names.sort_values(by="classifier", ascending=False).classifier[:get_best]
        ensemble_clf = np.array(self.nodes)[np.array(clf_to_choose.index)]
        ensemble_names = np.array(self.node_names)[np.array(clf_to_choose.index)]

        self.nodes = list(ensemble_clf)
        self.node_names = list(ensemble_names)

    def fit(self, df, target_variable, subject_df, subject_column, split_fraction=0.8, fit_variables=None,
            n_train_batches=None, n_test_batches=None, train_batch_size=None, test_batch_size=None, score_cap=0.5,
            get_best=None, verbose=False):
        """
        Fits the ensemble with train and test batches by randomly picking couples and fitting a weak classifier from
        a list provided by the user.

        Args:
            df: pandas.DataFrame with signal data.
            target_variable: Target variable to fit the ensemble.
            subject_df: pandas.DataFrame with subject data.
            subject_column: subject ID columns in subject_df.
            split_fraction: Train/test sampling in batching process.
            fit_variables: Variable to consider upon fit.
            n_train_batches: Number of training batches.
            n_test_batches: Number of test batches.
            train_batch_size: Size of training batches.
            test_batch_size: Size of test batches.
            score_cap: Minimum accuracy of a weak classifier to be eligible.
            get_best: The number of best weak classifiers to keep.
            verbose: Whether to print progress on screen.

        """

        batch_list_train, batch_list_test = self._prepare_batches(df=df,
                                                                  target_variable=target_variable,
                                                                  subject_df=subject_df,
                                                                  subject_column=subject_column,
                                                                  split_fraction=split_fraction,
                                                                  fit_variables=fit_variables,
                                                                  n_train_batches=n_train_batches,
                                                                  n_test_batches=n_test_batches,
                                                                  train_batch_size=train_batch_size,
                                                                  test_batch_size=test_batch_size,
                                                                  verbose=verbose)

        self._check_batches(batch_list_train, batch_list_test)

        print("---Fitting...")
        empty_string = " "

        m = 1
        for clf, length, args in zip(self.classifier_list, self.node_sizes, self.kwargs_list):
            n = 1
            while n <= length:

                ind_train = np.random.choice(batch_list_train.n_batches)
                ind_test = np.random.choice(batch_list_test.n_batches)

                x_train = batch_list_train.extract_batch(ind_train)
                y_train = batch_list_train.extract_batch(ind_train, y=True)

                x_test = batch_list_train.extract_batch(ind_test)
                y_test = batch_list_train.extract_batch(ind_test, y=True)

                _save_args = args.copy()

                for key in args.keys():
                    if type(args[key]) == tuple and len(args[key]) == 2:
                        if type(args[key][0]) == int:
                            args[key] = np.random.randint(args[key][0], args[key][1])
                        else:
                            args[key] = np.random.uniform(args[key][0], args[key][1])

                _clf = clf(**args)
                _clf.fit(x_train, y_train)

                args = _save_args

                train_score = _clf.score(x_train, y_train)
                test_score = _clf.score(x_test, y_test)

                if test_score >= score_cap:

                    if verbose:
                        to_print = f"node_test_acc_{test_score}_train_acc_{train_score}"
                        print(m, to_print, empty_string * (108 - len(to_print)),
                              flush=True, end="\r")

                    self.node_names.append([m, f"node_test_acc_{test_score}_train_acc_{train_score}"])
                    self.nodes.append(_clf)
                    m += 1
                    n += 1

            print(empty_string * 108, flush=True, end="\r")

        if get_best is not None:
            self._filter_nodes(get_best=get_best)

        print("---Fitting complete.                                                                        ")

    def predict_proba(self, df, verbose=False):
        """
        Return the probabilities that the instances of a batch of data stored in a pandas.DataFrame belong to the
        positive class. This data frame such contain compatible variable names. Note that the user will still have to
        make sure that such names refer to the right variables.

        Args:
            df: pandas.DataFrame with data to predict.
            verbose: Whether to print progress on screen.

        Returns:
            numpy.array: prediction probabilities.

        """

        assert self.ensemble_target_variable is not None, "Ensemble not fitted yet."
        assert type(df) == pd.DataFrame, "df expects a pandas.DataFrame."
        assert all([var in list(df.columns) for var in self.ensemble_input_variables]), "Data frame has mismatching" \
                                                                                        "input variables."

        predictions = []
        for k, node in enumerate(self.nodes):
            if verbose:
                print(f"Prediction with node {k + 1} of {len(self.nodes)}...", flush=True, end="\r")
            predictions.append(list(node.predict(np.array(df.loc[:, self.ensemble_input_variables]))))

        predictions = np.array(predictions).T

        if verbose:
            print("Predict completed.                                                                     ", flush=True)

        return np.mean(predictions, axis=1)

    def predict(self, df, threshold=0.5, verbose=False):
        """
        Classifies the instances of a batch of data contained in pandas.DataFrame into the positive or negative classes
        according to a certain probability threshold.

        Args:
            df: pandas.DataFrame containing the batch of data to be classified.
            threshold: Float between 0 and 1 stating the probability threshold to label an instance as positive.
            verbose: Whether to print progress on screen.

        Returns:
            numpy.array: class labels.

        """

        assert 0. < threshold < 1., "threshold expects a float between 0 and 1."

        predictions = self.predict_proba(df=df, verbose=verbose)

        return (predictions > threshold) * 1
