from importlib import import_module

VALID_APPLY_FUNCTIONS = {
    "mean": "numpy.mean",
    "median": "numpy.median",
    "std": "numpy.std",
    "sum": "numpy.sum"
}


def get_apply_functions(apply_function_list):
    """
    Imports the necessary module for each potential data set transformer stored in a list. Such models shall be among the
    ones present in VALID_TRANSFORMER_CLASSES.
    Args:
        apply_function_list: A list containing the objects to import, represented by their actual names as strings.

    Returns:
        An iterable that sequentially imports the needed module for each object.

    """

    assert all([trf in VALID_APPLY_FUNCTIONS.keys() for trf in apply_function_list]), "Found some non-compatible " \
                                                                                      "estimators in apply function " \
                                                                                      "list."

    for trf in apply_function_list:
        module = ".".join(VALID_APPLY_FUNCTIONS[trf].split(".")[:-1])
        obj = VALID_APPLY_FUNCTIONS[trf].split(".")[-1]
        lib = import_module(name=module)
        trf_object = getattr(lib, obj)

        yield trf_object
