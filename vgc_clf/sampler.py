import json
import numpy as np
import os
import pickle


class Sampler:

    def __init__(self):

        self.df = None
        self.input_variables = None
        self.target_variable = None
        self.batches = {}
        self.n_batches = 0
        self.balanced = None
        self.resample = None

    def save_sampling(self, path, sampling_name, mode="pickle"):

        assert self.balanced is not None and self.resample is not None, "No sampling has been done yet."

        if not os.path.exists(path):
            os.mkdir(path)

        output_dictionary = {
            "input_variables": self.input_variables,
            "target_variable": self.target_variable,
            "balanced": self.balanced,
            "resample": self.resample,
            "batches": self.batches
        }

        if mode == "pickle":
            pickle.dump(output_dictionary, open(f"{path}/{sampling_name}.pkl", "wb"))
        elif mode == "json":
            json.dump(output_dictionary, open(f"{path}/{sampling_name}.json", "w"))
        else:
            raise Exception(f"File save in format {mode} not implemented yet.")

    def load_sampling(self, path, sampling_name):

        assert os.path.exists(path), f"Directory {path} not found."

        path_contents = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        if sampling_name + ".pkl" in path_contents:
            input_dictionary = pickle.load(open(f"{path}/{sampling_name}.pkl", "rb"))
        elif sampling_name + ".json" in path_contents:
            input_dictionary = json.load(open(f"{path}/{sampling_name}.json", "r"))
        else:
            raise Exception(f"No valid file formats found in {path}")

        self.input_variables = input_dictionary["input_variables"]
        self.target_variable = input_dictionary["target_variable"]
        self.balanced = input_dictionary["balanced"]
        self.resample = input_dictionary["resample"]
        self.batches = input_dictionary["batches"]
        self.n_batches = len(input_dictionary["batches"].keys())

    def _get_data_frame_info(self, df, target_variable, input_variables=None):

        assert target_variable in list(df.columns), "Target variable not present in data frame columns."

        if input_variables is None:
            input_variables = list(df.columns)

        if target_variable in input_variables:
            input_variables.remove(target_variable)

        assert all([var in list(df.columns) for var in input_variables]), "Some input variables are not present in" \
                                                                          "data frame columns."

        self.df = df
        self.input_variables = input_variables
        self.target_variable = target_variable

    def generate_batches(self, df, n_batches, batch_len, target_variable, input_variables=None, balanced=True,
                         resample=True, verbose=True):

        self._get_data_frame_info(df=df, target_variable=target_variable, input_variables=input_variables)

        self.balanced = balanced
        self.resample = resample

        for i in np.arange(n_batches):

            if verbose:
                print(f"Working on batch {i + 1} of {n_batches}...", flush=True)

            if balanced:
                _df_0 = self.df[self.df[self.target_variable] == 0]
                _df_1 = self.df[self.df[self.target_variable] == 1]

                _s0 = np.random.choice(_df_0.index, size=int(batch_len / 2), replace=True)
                _s1 = np.random.choice(_df_1.index, size=int(batch_len / 2), replace=True)

                _sX = self.df.loc[np.hstack((_s0, _s1)), self.input_variables]
                _sy = self.df.loc[np.hstack((_s0, _s1)), self.target_variable]
            else:
                _s0 = np.random.choice(self.df.index, size=batch_len, replace=resample)

                _sX = self.df.loc[_s0, self.input_variables]
                _sy = self.df.loc[_s0, self.target_variable]

            _batch = {
                "X": np.array(_sX),
                "y": np.array(_sy)
            }

            _batches = {
                "batch_" + str(i): _batch
            }

            self.batches.update(_batches)

        self.n_batches = len(self.batches.keys())

        print("Batch sampling complete.", flush=True)

    def extract_batch(self, index, y=False):

        assert index < self.n_batches, "Given index is out of bounds"

        to_extract = self.batches["batch_" + str(index)]

        if y:
            return to_extract["y"]
        else:
            return to_extract["X"]

    def combine_samplings(self, new_sampling):

        assert type(new_sampling) == Sampler
        assert new_sampling.target_variable == self.target_variable, "Target variable names do not match."
        assert len(new_sampling.input_variables) == len(self.input_variables), "Input variables length mismatch."
        assert all([var in self.input_variables for var in new_sampling.input_variables]), "Input variable mismatch"
        assert all([var in new_sampling.input_variables for var in self.input_variables]), "Input variable mismatch"

        k = self.n_batches

        for key in new_sampling.batches.keys():
            self.batches[f"batch_{k}"] = new_sampling.batches[key]
            k += 1

        self.n_batches = len(self.batches.keys())

        if self.balanced and not new_sampling.balanced:
            self.balanced = False

        if not self.resample and new_sampling.resample:
            self.resample = True

    def __repr__(self):

        if self.balanced is None:
            print("Empty Sampler object.")
        else:
            out_string = f"Sampler object consisting of {self.n_batches} "
            if self.balanced:
                out_string += "balanced "
            out_string += f"batches "
            if self.resample:
                out_string += "with resampling "
            out_string += f"generated with target variable {self.target_variable} and input variables" \
                          f" {self.input_variables}."

            print(out_string)
