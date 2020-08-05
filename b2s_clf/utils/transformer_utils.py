from importlib import import_module

VALID_TRANSFORMER_CLASSES = {
    "one_hot": "category_encoders.OneHotEncoder",
    "target_encoder": "category_encoders.TargetEncoder",
    "min_max_normalization": "sklearn.preprocessing.MinMaxScaler",
    "mean_var_normalization": "sklearn.preprocessing.StandardScaler",
}


def get_transformer_objects(transformer_list):
    """
    Imports the necessary module for each potential data set transformer stored in a list. Such models shall be among the
    ones present in VALID_TRANSFORMER_CLASSES.

    Args:
        transformer_list: A list containing the objects to import, represented by their actual names as strings.

    Returns:
        Iterable: Sequentially imports the needed module for each object.

    """

    assert all([trf in VALID_TRANSFORMER_CLASSES.keys() for trf in transformer_list]), "Found some non-compatible " \
                                                                                       "transformers in transformer " \
                                                                                       "list."

    for trf in transformer_list:
        module = ".".join(VALID_TRANSFORMER_CLASSES[trf].split(".")[:-1])
        obj = VALID_TRANSFORMER_CLASSES[trf].split(".")[-1]
        lib = import_module(name=module)
        trf_object = getattr(lib, obj)

        yield trf_object
