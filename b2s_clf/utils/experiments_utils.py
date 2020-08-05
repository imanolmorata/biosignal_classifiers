import numpy as np

from b2s_clf.dataset_transformer.encoder import Encoder
from b2s_clf.dataset_transformer.normalizer import Normalizer
from b2s_clf.dataset_transformer.signal_compressor import SignalCompressor as sg_com


def transform_with_encoders(df, df_fit, df_val, valid_variables, encoder_list, encoder_kwargs, encoders_input_columns,
                            encoders_target_columns, verbose=False):
    """
    Transforms a train and test data frame according to a series of category_encoders.* objects

    Args:
        df: pandas.DataFrame with signal data.
        df_fit: pandas.DataFrame with training data.
        df_val: pandas.DataFrame with test data.
        valid_variables: List of variables present in df_fit and df_val that should be used.
        encoder_list: List of category_encoder objects to apply.
        encoder_kwargs: List of additional arguments for each encoder object.
        encoders_input_columns: List of lists of variables that correspond to each encoder step.
        encoders_target_columns: List of target variables that correspond to each encoder step.
        verbose: Whether to print progress on screen.

    Returns:
        pandas.DataFrame: Transformed training set.
        pandas.DataFrame: Transformed test set.
        list: Transformed variable names.

    """
    print("ENCODER", flush=True)
    encoder_obj = Encoder(transformer_list=encoder_list,
                          kwargs_list=encoder_kwargs,
                          input_cols_list=encoders_input_columns,
                          target_col_list=encoders_target_columns)
    df_fit = encoder_obj.fit_transform(df=df_fit, verbose=verbose)
    df_val = encoder_obj.transform(df=df_val, verbose=verbose)

    for var in list(np.array(encoders_input_columns).ravel()):
        if var in valid_variables:
            valid_variables.remove(var)

    for var in list(df_fit.columns):
        if var not in list(df.columns):
            valid_variables.append(var)

    return df_fit, df_val, valid_variables


def transform_with_normalizers(df_fit, df_val, normalizers_list, normalizers_kwargs, normalizers_input_columns,
                               verbose=False):
    """
    Transforms a train and test data frame according to a series of sklearn.preprocessing normalizer objects.

    Args:
        df_fit: pandas.DataFrame with training data.
        df_val: pandas.DataFrame with test data.
        normalizers_list: A list of sklearn.preprocessing objects.
        normalizers_kwargs: Additional kwargs for normalizer objects.
        normalizers_input_columns: List of lists of variables to apply each normalizer to.
        verbose: Whether to print progress on screen.

    Returns:
        pandas.DataFrame: Transformed training set.
        pandas.DataFrame: Transformed test set.

    """
    print("NORMALIZER", flush=True)
    normalizer_obj = Normalizer(transformer_list=normalizers_list,
                                kwargs_list=normalizers_kwargs,
                                input_cols_list=normalizers_input_columns)
    df_fit = normalizer_obj.fit_transform(df=df_fit, verbose=verbose)
    df_val = normalizer_obj.transform(df=df_val, verbose=verbose)

    return df_fit, df_val


def transform_with_signal_compressors(df_fit, df_val, valid_variables, signal_compressor_clusters,
                                      signal_compressor_input_columns, signal_compressor_apply_functions,
                                      verbose=False):
    """
    Transforms a data set containing signal data by compressing such signals.

    Args:
        df_fit: pandas.DataFrame with training data.
        df_val: pandas.DataFrame with test data.
        valid_variables: List of variables present in df_fit and df_val that should be used.
        signal_compressor_clusters: Number of signal chunks to compress.
        signal_compressor_input_columns: List of variables that reference the signal data.
        signal_compressor_apply_functions: List of callables to transform the chunks into single values.
        verbose: Whether to print progress on screen.

    Returns:
        pandas.DataFrame: Transformed training set.
        pandas.DataFrame: Transformed test set.
        list: Transformed variable names.

    """
    print("COMPRESSOR", flush=True)
    compressor_obj = sg_com(n_clusters_list=signal_compressor_clusters,
                            input_cols_list=signal_compressor_input_columns,
                            apply_estimator_list=signal_compressor_apply_functions)
    compressor_obj.fit(df=df_fit, verbose=verbose)
    df_fit = compressor_obj.transform(df=df_fit, verbose=verbose)
    df_val = compressor_obj.transform(df=df_val, verbose=verbose)

    for var in list(np.array(signal_compressor_input_columns).ravel()):
        if var in valid_variables:
            valid_variables.remove(var)

    for var in list(df_fit.columns):
        if "compressed_" in var and "frame_" in var:
            valid_variables.append(var)

    return df_fit, df_val, valid_variables
