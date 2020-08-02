from importlib import import_module

VALID_CLASSIFIER_CLASSES = {
    "logistic": "sklearn.linear_model.LogisticRegression",
    "Ridge": "sklearn.linear_model.RidgeClassifier",
    "SVM": "sklearn.svm.SVC",
    "tree": "sklearn.tree.DecisionTreeClassifier",
    "rf": "sklearn.ensemble.RandomForestClassifier",
    "knn": "sklearn.neighbors.KNeighborsClassifier"
}


def get_classifier_objects(classifier_list):

    assert all([clf in VALID_CLASSIFIER_CLASSES.keys() for clf in classifier_list]), "Found some non-compatible " \
                                                                                     "classifiers in classifier list."

    for clf in classifier_list:
        module = ".".join(VALID_CLASSIFIER_CLASSES[clf].split(".")[:-1])
        obj = VALID_CLASSIFIER_CLASSES[clf].split(".")[-1]
        lib = import_module(name=module)
        clf_object = getattr(lib, obj)

        yield clf_object
