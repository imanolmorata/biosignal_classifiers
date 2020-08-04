from vgc_clf.dataset_transformer.dataset_transformer import DatasetTransformer

VALID_ENCODERS = ["OneHotEncoder", "TargetEncoder", "type"]


class Encoder(DatasetTransformer):

    def __init__(self, transformer_list, kwargs_list, input_cols_list, target_col_list):

        super().__init__(transformer_list, kwargs_list, input_cols_list)

        self.target_col_list = target_col_list

        self._add_input_cols_to_kwargs()
        self._check_transformers_are_encoders()

    def _add_input_cols_to_kwargs(self):
        """
        Input column names are only needed in encoder transformation, thus are added to encoder kwargs list.
        Returns:

        """

        for kwargs, col_list in zip(self.kwargs_list, self.input_cols_list):
            kwargs["cols"] = col_list

    def _check_transformers_are_encoders(self):
        """
        Checks if whether the self.transfomer_list contains only category encoders.
        Returns:

        """

        assert all([trf.__class__.__name__ in VALID_ENCODERS for trf in self.transformer_list]), "Non-encoder " \
                                                                                                 "objects present in " \
                                                                                                 "transformer list."

    def fit(self, df, verbose=False):
        """
        Fits all categorical encoders in present in self.transformer_list.
        Args:
            df: pandas.DataFrame with fit data.
            verbose: Whether to print progress on screen.

        Returns:

        """

        self.fitted_objects = []
        for k, trf in enumerate(self.transformer_list):

            if verbose:
                print(f"Fitting encoder {k + 1} of {len(self.transformer_list)}...              ", flush=True, end="\r")

            trf_obj = trf(**self.kwargs_list[k])
            if self.target_col_list[k] is None:
                y_fit = None
            else:
                y_fit = df[self.target_col_list[k]]
            trf_obj.fit(df, y=y_fit)
            self.fitted_objects.append(trf_obj)

        if verbose:
            print("---Fit complete.                               ", flush=True)

    def fit_transform(self, df, verbose=False):
        """
        Fits all categorical encoders in present in self.transformer_list and return a transformation of the training
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
                print(f"Fitting encoder {k + 1} of {len(self.transformer_list)}...              ", flush=True, end="\r")

            trf_obj = trf(**self.kwargs_list[k])
            if self.target_col_list[k] is None:
                y_fit = None
            else:
                y_fit = df[self.target_col_list[k]]
            trf_obj.fit(df, y=y_fit)
            self.fitted_objects.append(trf_obj)

            df = trf_obj.transform(df)

        if verbose:
            print("---Fit complete.                               ", flush=True)

        return df

    def transform(self, df, verbose=False):
        """
        Transforms data using all encoders fitted in self.fitted_objects.
        Args:
            df: pandas.DataFrame with data to transform.
            verbose: Whether to print progress on screen.

        Returns:
            df: Transformed data set.

        """

        assert len(self.fitted_objects) == len(self.transformer_list), "Encoder not fitted yet or wrong fit."

        for k, trf_obj in enumerate(self.fitted_objects):

            if verbose:
                print(f"Transforming with encoder {k + 1} of {len(self.transformer_list)}...              ", flush=True,
                      end="\r")

            df = trf_obj.transform(df)

        if verbose:
            print("---Transform complete.                        ", flush=True)

        return df
