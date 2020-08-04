import numpy as np

from b2s_clf.dataset_transformer.encoder import Encoder
from b2s_clf.dataset_transformer.normalizer import Normalizer
from b2s_clf.dataset_transformer.signal_compressor import SignalCompressor as sg_com


def transform_with_encoders(df, df_fit, df_val, valid_variables, encoder_list, encoder_kwargs, encoders_input_columns,
                            encoders_target_columns, verbose=False):
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
