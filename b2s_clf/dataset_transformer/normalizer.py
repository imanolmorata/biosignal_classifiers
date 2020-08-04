from b2s_clf.dataset_transformer.dataset_transformer import DatasetTransformer

VALID_NORMALIZERS = ["MinMaxScaler", "StandardScaler", "type"]


class Normalizer(DatasetTransformer):

    def __init__(self, transformer_list, kwargs_list, input_cols_list):

        super().__init__(transformer_list, kwargs_list, input_cols_list)

        self._check_transformers_are_normalizers()

    def _check_transformers_are_normalizers(self):
        """
        Checks if whether the self.transformer_list contains only normalizers.
        Returns:

        """

        assert all([trf.__class__.__name__ in VALID_NORMALIZERS for trf in self.transformer_list]), "Non-normalizer " \
                                                                                                    "objects present " \
                                                                                                    "in transformer " \
                                                                                                    "list."

    def fit(self, df, verbose=False):
        """
        Fits all normalizers in present in self.transformer_list.
        Args:
            df: pandas.DataFrame with fit data.
            verbose: Whether to print progress on screen.

        Returns:

        """

        self.fitted_objects = []
        for k, trf in enumerate(self.transformer_list):

            if verbose:
                print(f"Fitting normalizer {k + 1} of {len(self.transformer_list)}...           ", flush=True, end="\r")

            trf_obj = trf(**self.kwargs_list[k])
            trf_obj.fit(df.loc[:, self.input_cols_list[k]])
            self.fitted_objects.append(trf_obj)

        if verbose:
            print("---Fit complete.                               ", flush=True)

    def fit_transform(self, df, verbose=False):
        """
        Fits all normalizers in present in self.transformer_list and return a transformation of the training
        set.
        Args:
            df: pandas.DataFrame with fit data.
            verbose: Whether to print progress on screen.

        Returns:
            df: Transformed training set.

        """

        self.fitted_objects = []
        for k, trf in enumerate(self.transformer_list):

            if verbose:
                print(f"Fitting normalizer {k + 1} of {len(self.transformer_list)}...           ", flush=True, end="\r")

            trf_obj = trf(**self.kwargs_list[k])
            trf_obj.fit(df.loc[:, self.input_cols_list[k]])
            self.fitted_objects.append(trf_obj)

            df.loc[:, self.input_cols_list[k]] = trf_obj.transform(df.loc[:, self.input_cols_list[k]])

        if verbose:
            print("---Fit complete.                               ", flush=True)

        return df

    def transform(self, df, verbose=False):
        """
        Transforms data using all normalizers fitted in self.fitted_objects.
        Args:
            df: pandas.DataFrame with data to transform.
            verbose: Whether to print progress on screen.

        Returns:
            df: Transformed data set.

        """

        assert len(self.fitted_objects) == len(self.transformer_list), "Normalizer not fitted yet or wrong fit."

        for k, trf_obj in enumerate(self.fitted_objects):

            if verbose:
                print(f"Transforming with normalizer {k + 1} of {len(self.transformer_list)}...           ", flush=True,
                      end="\r")

            df.loc[:, self.input_cols_list[k]] = trf_obj.transform(df.loc[:, self.input_cols_list[k]])

        if verbose:
            print("---Transform complete.                        ", flush=True)

        return df
