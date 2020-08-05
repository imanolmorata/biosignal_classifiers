import numpy as np
import pandas as pd

from sklearn.cluster import KMeans


class SignalCompressor:
    """
    A class that performs compression on signal data.
    """

    def __init__(self, n_clusters_list, input_cols_list, apply_estimator_list):

        self.n_clusters_list = n_clusters_list
        self.input_cols_list = input_cols_list
        self.apply_estimator_list = apply_estimator_list
        self.signal_clusters = None
        self.compressed_signal_column_names = {}

    def fit(self, df, verbose=False):
        """
        Fits clusters according to variable groups present in self.input_cols_list. These will be used to compress
        signal data.

        Args:
            df: pandas.DataFrame with signal data.
            verbose: Whether to print progress on screen.

        """

        self.signal_clusters = []
        for k, n_clus in enumerate(self.n_clusters_list):

            if verbose:
                print(f"Compressing signals {k + 1} of {len(self.n_clusters_list)}...           ", flush=True, end="\r")

            x = np.array(df[self.input_cols_list[k]].T)
            cluster_object = KMeans(n_clusters=n_clus).fit(x)
            km_labels = cluster_object.labels_

            m = -1
            col_groups = []
            for label in km_labels:

                if label == m:
                    continue

                col_group = list(np.array(self.input_cols_list[k])[km_labels == label])
                col_groups.append(col_group)

                m = label

            self.signal_clusters.append(col_groups)

        if verbose:
            print("---Fit complete.                               ", flush=True)

    def transform(self, df, verbose=False):
        """
        Compresses signal data using all clusters fitted in self.signal_clusters.

        Args:
            df: pandas.DataFrame with signal data to compress.
            verbose: Whether to print progress on screen.

        Returns:
            df: Compressed signal data set.

        """

        assert self.signal_clusters is not None, "Compressor not fitted yet."

        for k, col_groups in enumerate(self.signal_clusters):

            if verbose:
                print(f"Transforming signals {k + 1} of {len(self.signal_clusters)}...           ", flush=True,
                      end="\r")

            signal_groups = []
            for col_group in col_groups:
                signal_groups.append(df[col_group].apply(self.apply_estimator_list[k], axis=1))

            df_compressed = pd.concat(signal_groups, axis=1)
            df_compressed.columns = [f"compressed_{k + 1}_frame_{n + 1}" for n in np.arange(self.n_clusters_list[k])]

            self.compressed_signal_column_names[f"compression_{k + 1}"] = {
                "original_names": list(self.input_cols_list[k]),
                "compression_names": list(df_compressed.columns)
            }
            df = pd.concat([df.drop(self.input_cols_list[k], axis=1), df_compressed], axis=1)

        if verbose:
            print("---Transform complete.                        ", flush=True)

        return df
