from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from ipaddress import ip_address
import pandas as pd
import numpy as np

def ip_features_generator(df, ip_column):
    n = len(df)
    df_master = pd.DataFrame()
    for col in ip_column:
        df_features = pd.DataFrame({f"{col}_host": np.zeros(n),
                                    f"{col}_broadcast": np.zeros(n),
                                    f"{col}_ipv6": np.zeros(n),
                                    f"{col}_privado": np.zeros(n),
                                    f"{col}_multicast": np.zeros(n)})

        ip_series = df[col].apply(ip_address)

        # Mascaras
        ip_host_mask = ip_series == ip_address('127.0.0.1')
        ip_broadcast_mask = ip_series == ip_address('255.255.255.255')
        ip_privado_mask = ip_series.apply(lambda x: x.is_private)
        ip_multicast_mask = ip_series.apply(lambda x: x.is_multicast)


        # Preenchimento das novas colunas
        df_features.loc[ip_host_mask, f'{col}_host'] = 1
        df_features.loc[ip_broadcast_mask, f'{col}_broadcast'] = 1
        df_features.loc[ip_series.apply(lambda x: x.version == 6), f'{col}_ipv6'] = 1
        df_features.loc[ip_privado_mask, f'{col}_privado'] = 1
        df_features.loc[ip_multicast_mask, f'{col}_multicast'] = 1

        df_master = pd.concat([df_master, df_features], axis=1)

    return df_master


def port_features_generator(df, port_column):
    n = len(df)
    df_master = pd.DataFrame()
    for col in port_column:
        df_features = pd.DataFrame({f"{col}_well_known": np.zeros(n),
                                    f"{col}_registered": np.zeros(n),
                                    f"{col}_dynamic": np.zeros(n)})

        port_series = df[col]

        # Mascaras
        port_well_known_mask = port_series < 1024
        port_registered_mask = (port_series >= 1024) & (port_series < 49152)
        port_dynamic_mask = port_series >= 49152

        # Preenchimento das novas colunas
        df_features.loc[port_well_known_mask, f'{col}_well_known'] = 1
        df_features.loc[port_registered_mask, f'{col}_registered'] = 1
        df_features.loc[port_dynamic_mask, f'{col}_dynamic'] = 1

        df_master = pd.concat([df_master, df_features], axis=1)

    return df_master


def remove_columns(X, columns_to_drop):
    return X.drop(columns=columns_to_drop, axis=1)


def build_preprocessing_pipeline(features_numericas,
                                 features_categoricas,
                                 features_textuais,
                                 features_ip,
                                 features_port):

    # Adicionando a transformação de IP
    ip_transformer = Pipeline(steps=[
        ('ip_features', FunctionTransformer(ip_features_generator, kw_args={'ip_column': features_ip}))])

    # Adicionando a transformação de porta
    port_transformer = Pipeline(steps=[
        ('port_features', FunctionTransformer(port_features_generator, kw_args={'port_column': features_port}))])

    # Pipeline para colunas numéricas (imputação e normalização)
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler()) # Normalização
    ])

    # Pipeline para colunas categóricas (imputação e codificação one-hot)
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='if_binary', sparse_output=False)) # Codificação one-hot
    ])

    textual_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='if_binary',
                                 min_frequency=0.1, sparse_output=False)) # Codificação one-hot
    ])

    # Combinando os transformers
    preprocessor = ColumnTransformer(transformers=[
        ('ip', ip_transformer, features_ip),
        ('port', port_transformer, features_port),
        ('num', numeric_transformer, features_numericas),
        ('cat', categorical_transformer, features_categoricas),
        ('text', textual_transformer, features_textuais)
    ],
        verbose_feature_names_out=False
    )

    # Construção do pipeline final com seleção de features e redução de dimensionalidade
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor)]).set_output(transform='pandas')

    return pipeline
