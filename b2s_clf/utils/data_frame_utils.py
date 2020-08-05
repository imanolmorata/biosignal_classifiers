import numpy as np
import pandas as pd


def get_subjects_data_frame(df, subject_column_name, subject_info_columns=None):
    """
    Extract subject data from a data frame that contains signals attached to subjects. This is useful in the event 
    that all subjects have more than one signal referenced to them.

    Args:
        df: pandas.DataFrame containing, at least, signal data and subject data.
        subject_column_name: Specific column with (unique) subject ID or naming logic.
        subject_info_columns: List of columns that contain subject info.

    Returns:
        pandas.DataFrame: Data frame with as many rows as unique subjects and those columns specified in
        subject_info_columns.

    """
    assert subject_column_name in list(df.columns), "subject column not found in data frame."

    if subject_info_columns is None:
        subject_info_columns = list(df.columns)
    if subject_column_name not in subject_info_columns:
        subject_info_columns = [subject_column_name] + subject_info_columns

    df_subjects = df[subject_info_columns].groupby(subject_column_name, as_index=False).max()

    return df_subjects


def get_balanced_sample_from_data_frame(df, balancing_variable, max_len=None):
    """
    Generates a balanced subsample of a given pandas.DataFrame according to one binary variable.

    Args:
        df: pandas.DataFrame containing some data.
        balancing_variable: Binary variable with respect to which the sampling shall be made.
        max_len: Maximum length per class in the balancing.

    Returns:
        pandas.DataFrame: Balanced subsample of df.

    """
    assert len(df[balancing_variable].unique()) == 2, "Passed balancing variable is not binary."

    ref_val = list(df[balancing_variable].unique())[0]

    if max_len is None:
        max_len = len(df)

    condition = df[balancing_variable] == ref_val

    allowed_len = np.min([max_len, len(df[condition]), len(df[~condition])])

    negative_index = np.random.choice(df[~condition].index, size=allowed_len)
    positive_index = np.random.choice(df[condition].index, size=allowed_len)

    balanced_df = pd.concat([df.loc[negative_index, :], df.loc[positive_index, :]])
    balanced_df = balanced_df.sample(frac=1.)

    return balanced_df


def get_train_validation_from_data_frame(signal_df, subjects_df, subject_id_column, target_variable, test_size):
    """
    Generates a train and validation pair from a data frame containing at least signal and subject data.
    Args:
        signal_df: pandas.DataFrame containing signal and subject data.
        subjects_df: pandas.DataFrame containing subject data, as in get_subjects_data_frame.
        subject_id_column: Column containing subject ID or naming logic.
        target_variable: Binary target variable.
        test_size: Size of test sample.

    Returns:
        pandas.DataFrame: Subsample of df with training data.
        pandas.DataFrame: Subsample of df with validation data.
        pandas.Index: Indices of subject_df used to get signal_fit.
        pandas.Index: Indices of subject_df used to get signal_val.

    """
    assert len(subjects_df[target_variable].unique()) == 2

    ref_val = list(subjects_df[target_variable].unique())[0]

    condition = subjects_df[target_variable] == ref_val
    max_len = np.min([len(subjects_df[~condition]), len(subjects_df[condition])])

    if test_size > max_len:
        test_size = max_len

    negative_index = np.random.choice(subjects_df[~condition].index, size=test_size)
    positive_index = np.random.choice(subjects_df[condition].index, size=test_size)

    index_val = pd.Index(list(negative_index) + list(positive_index))
    index_fit = pd.Index(set(subjects_df.index) - set(index_val))

    signal_val = signal_df.merge(subjects_df.loc[index_val, subject_id_column], on=subject_id_column, how="right")
    signal_fit = signal_df.merge(subjects_df.loc[index_fit, subject_id_column], on=subject_id_column, how="right")

    return signal_fit, signal_val, index_fit, index_val


def get_strata_samples_from_data_frame(signal_df, subjects_df, subject_id_column, strata_column, strata_value,
                                       balanced_by=None):
    """
    Generates a stratified pair of sub-samples from a data frame containing at least signal and subject data. The pair
    is generated in a leave-one-out fashion, with one containing only one strata and the other containing the rest.

    Args:
        signal_df: pandas.DataFrame containing signal and subject data.
        subjects_df: pandas.DataFrame containing subject data, as in get_subjects_data_frame.
        subject_id_column: Column containing subject ID or naming logic.
        strata_column: Stratification variable.
        strata_value: The particular strata to be the one-left-out.
        balanced_by: Whether to balance the subsample by a second binary variable.

    Returns:
        pandas.DataFrame: Subsample of df without strata_value.
        pandas.DataFrame: Subsample of df with only strata_value.
        pandas.Index: Indices of subject_df used to get signal_neg.
        pandas.Index: Indices of subject_df used to get signal_pos.

    """
    assert strata_value in list(subjects_df[strata_column].unique()), "strata_value not present in strata column."

    condition = subjects_df[strata_column] == strata_value

    negative_index = subjects_df[~condition].index
    positive_index = subjects_df[condition].index

    if balanced_by is not None:
        check_neg = len(subjects_df.loc[negative_index, balanced_by].unique()) == 2
        check_pos = len(subjects_df.loc[positive_index, balanced_by].unique()) == 2

        if check_neg:
            negative_index = get_balanced_sample_from_data_frame(subjects_df.loc[negative_index, :],
                                                                 balancing_variable=balanced_by).index
        if check_pos:
            positive_index = get_balanced_sample_from_data_frame(subjects_df.loc[positive_index, :],
                                                                 balancing_variable=balanced_by).index

    signal_neg = signal_df.merge(subjects_df.loc[negative_index, subject_id_column], on=subject_id_column, how="right")
    signal_pos = signal_df.merge(subjects_df.loc[positive_index, subject_id_column], on=subject_id_column, how="right")

    return signal_neg, signal_pos, negative_index, positive_index


def generate_cross_validation_batch(n_batches, signal_df, subjects_df, subject_id_column, target_variable, test_size):
    """
    Generates an iterable of train/test samples from a given pandas.DataFrame containing at least signal and subject
    data. It is a lazy generator, meaning that each instance of the iterator is given by the yield method rather than
    storing all instances in memory and then running over them.

    Args:
        n_batches: How many batches to generate.
        signal_df: pandas.DataFrame containing signal and subject data.
        subjects_df: pandas.DataFrame containing subject data, as in get_subjects_data_frame.
        subject_id_column: Column containing subject ID or naming logic.
        target_variable: Binary target variable.
        test_size: Size of test sample.

    Returns:
        Iterable: Tetrads train_sample, test_sample, train_index, test_index

    """
    for _ in np.arange(n_batches):
        yield get_train_validation_from_data_frame(signal_df=signal_df,
                                                   subjects_df=subjects_df,
                                                   subject_id_column=subject_id_column,
                                                   target_variable=target_variable,
                                                   test_size=test_size)


def generate_leave_one_out_batch(signal_df, subjects_df, subject_id_column, strata_column, balanced_by=None):
    """
        Generates an iterable of has-strata/lacks-strata samples from a given pandas.DataFrame containing at least
        signal and subject data. It is a lazy generator, meaning that each instance of the iterator is given by the
        yield method rather than storing all instances in memory and then running over them.

        Args:
            signal_df: pandas.DataFrame containing signal and subject data.
            subjects_df: pandas.DataFrame containing subject data, as in get_subjects_data_frame.
            subject_id_column: Column containing subject ID or naming logic.
            strata_column: Stratification variable name in signal_df.
            balanced_by: [Optional] Whether to balance outcomes by a second binary variable.

        Returns:
            Iterable: Tetrads has_strata_sample, lacks_strata_sample, has_strata_index, lacks_strata_index

        """
    for strata in subjects_df[strata_column].unique():
        yield get_strata_samples_from_data_frame(signal_df=signal_df,
                                                 subjects_df=subjects_df,
                                                 subject_id_column=subject_id_column,
                                                 strata_column=strata_column,
                                                 strata_value=strata,
                                                 balanced_by=balanced_by)
