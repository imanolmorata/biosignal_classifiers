import numpy as np
import pandas as pd

from vgc_clf.sampler import Sampler


class Ensemble:

    def __init__(self, classifier_list, node_sizes, kwargs_list):

        self.classifier_list = classifier_list
        self.node_sizes = node_sizes
        self.kwargs_list = kwargs_list
        self.nodes = []
        self.node_names = []

    def _filter_nodes(self, get_best=100):

        clf_names = pd.DataFrame(self.node_names, columns=["idx", "classifier"])

        clf_to_choose = clf_names.sort_values(by="classifier", ascending=False).classifier[:get_best]
        ensemble_clf = np.array(self.nodes)[np.array(clf_to_choose.index)]
        ensemble_names = np.array(self.node_names)[np.array(clf_to_choose.index)]

        self.nodes = list(ensemble_clf)
        self.node_names = list(ensemble_names)

    def fit(self, batch_list_train, batch_list_test, score_cap=0.5, get_best=None, verbose=False):

        assert type(batch_list_train) == Sampler
        assert type(batch_list_test) == Sampler

        print("---Fitting...")

        l = 1
        for clf, length, args in zip(self.classifier_list, self.node_sizes, self.kwargs_list):
            n = 1
            while n <= length:

                ind_train = np.random.choice(len(batch_list_train))
                ind_test = np.random.choice(len(batch_list_test))

                x_train = batch_list_train.extract_btch(ind_train)
                y_train = batch_list_train.extract_btch(ind_train, y=True)

                x_test = batch_list_train.extract_btch(ind_test)
                y_test = batch_list_train.extract_btch(ind_test, y=True)

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
                        print(l, f"node_test_acc_{test_score}_train_acc_{train_score}", flush=True)

                    self.node_names.append([l, f"node_test_acc_{test_score}_train_acc_{train_score}"])
                    self.nodes.append(_clf)
                    l += 1
                    n += 1

        if get_best is not None:
            self._filter_nodes(get_best=get_best)

        print("---Fitting complete.")

    def predict(self, df, threshold=0.5):

        predictions = []
        for node in self.nodes:
            predictions.append(list(node.predict(np.array(df))))

        predictions = np.array(predictions).T

        return (np.mean(predictions, axis=0) > threshold) * 1
