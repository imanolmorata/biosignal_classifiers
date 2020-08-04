class DatasetTransformer:

    def __init__(self, transformer_list, kwargs_list, input_cols_list):

        self.transformer_list = transformer_list
        self.kwargs_list = kwargs_list
        self.input_cols_list = input_cols_list
        self.fitted_objects = []

        self._check_transformers()

    def _check_transformers(self):

        assert all([hasattr(trf, "fit") for trf in self.transformer_list]), "At least one transformer object is not " \
                                                                            "compatible with 'fit' method."
        assert all([hasattr(trf, "transform") for trf in self.transformer_list]), "At least one classifier object " \
                                                                                  "is not compatible with " \
                                                                                  "'transform' method."
        assert all([hasattr(trf, "fit_transform") for trf in self.transformer_list]), "At least one classifier " \
                                                                                      "object is not compatible with " \
                                                                                      "'fit_transform' method."
