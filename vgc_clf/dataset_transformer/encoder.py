from vgc_clf.dataset_transformer.dataset_transformer import DatasetTransformer


class Encoder(DatasetTransformer):

    def __init__(self, transformer_list, kwargs_list, input_cols_list, target_col_list):

        super().__init__(transformer_list, kwargs_list, input_cols_list)

        self.target_col_list = target_col_list
        self._add_input_cols_to_kwargs()

    def _add_input_cols_to_kwargs(self):

        for kwargs, col_list in zip(self.kwargs_list, self.input_cols_list):
            kwargs["cols"] = col_list

    def fit(self, df):

        self.fitted_objects = []
        for k, trf in enumerate(self.transformer_list):
            trf_obj = trf(**self.kwargs_list[k])
            trf_obj.fit(df, y=df[self.target_col_list[k]])
            self.fitted_objects.append(trf_obj)

    def fit_transform(self, df):

        self.fitted_objects = []
        for k, trf in enumerate(self.transformer_list):
            trf_obj = trf(**self.kwargs_list[k])
            trf_obj.fit(df, y=df[self.target_col_list[k]])
            self.fitted_objects.append(trf_obj)

            df = trf_obj.transform(df)

        return df

    def transform(self, df):

        assert len(self.fitted_objects) == len(self.transformer_list), "Encoder not fitted yet or wrong fit."

        for trf_obj in self.fitted_objects:
            df = trf_obj.transform(df)

        return df
