#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import dask
from itertools import product
from copy import deepcopy
from sklearn.utils.validation import check_is_fitted


class MetaModels:
    """
    Train models to ensemble.

    Parameters
    ----------
    models: dict
        Dictionary of models to train. Ej {'ARIMA': ARIMA()}
    scheduler: str
        Dask scheduler. See https://docs.dask.org/en/latest/setup/single-machine.html
        for details.
        Using "threads" can cause severe conflicts.
    """

    def __init__(self, models, scheduler='processes'):
        self.models = models
        self.scheduler = scheduler

    def fit(self, y_panel_df):
        """For each time series fit each model in models.

        y_panel_df: pandas df
            Pandas DataFrame with columns ['unique_id', 'ds', 'y']
        """

        fitted_models = []
        uids = []
        name_models = []
        for ts, meta_model in product(y_panel_df.groupby('unique_id'), self.models.items()):
            uid, y = ts
            y = y['y'].values
            name_model, model = deepcopy(meta_model)
            fitted_model = dask.delayed(model.fit)(y)
            fitted_models.append(fitted_model)
            uids.append(uid)
            name_models.append(name_model)

        fitted_models = dask.delayed(fitted_models).compute(scheduler=self.scheduler)

        fitted_models = pd.DataFrame.from_dict({'unique_id': uids,
                                                'model': name_models,
                                                'fitted_model': fitted_models})

        self.fitted_models_ = fitted_models.set_index(['unique_id', 'model'])

        return self

    def predict(self, y_hat_df):
        """Predict each model for each time series.

        y_hat_df: pandas df
            Pandas DataFrame with columns ['unique_id', 'ds']
        """
        check_is_fitted(self, 'fitted_models_')

        y_hat_df = deepcopy(y_hat_df[['unique_id', 'ds']])

        forecasts = []
        uids = []
        dss = []
        name_models = []
        for ts, name_model in product(y_hat_df.groupby('unique_id'), self.models.keys()):
            uid, df = ts
            h = len(df)
            model = self.fitted_models_.loc[(uid, name_model)]
            model = model.item()
            y_hat = dask.delayed(model.predict)(h)
            forecasts.append(y_hat)
            uids.append(np.repeat(uid, h))
            dss.append(df['ds'])
            name_models.append(np.repeat(name_model, h))

        forecasts = dask.delayed(forecasts).compute(scheduler=self.scheduler)
        forecasts = zip(uids, dss, name_models, forecasts)

        forecasts_df = []
        for uid, ds, name_model, forecast in forecasts:
            dict_df = {'unique_id': uid,
                       'ds': ds,
                       'model': name_model,
                       'forecast': forecast}
            df = dask.delayed(pd.DataFrame.from_dict)(dict_df)
            forecasts_df.append(df)

        forecasts = dask.delayed(forecasts_df).compute()
        forecasts = pd.concat(forecasts)

        forecasts = forecasts.set_index(['unique_id', 'ds', 'model']).unstack()
        forecasts = forecasts.droplevel(0, 1).reset_index()
        forecasts.columns.name = ''


        return forecasts

################################################################################
########## UTILS FOR FFORMA FLOW
###############################################################################

def temp_holdout(y_panel_df, val_periods):
    """Splits the data in train and validation sets.

    Parameters
    ----------
    y_panel_df: pandas df
        Pandas DataFrame with columns ['unique_id', 'ds', 'y']

    Returns
    -------
    Tuple
        - train: pandas df
        - val: pandas df
    """
    val = y_panel_df.groupby('unique_id').tail(val_periods)
    train = y_panel_df.groupby('unique_id').apply(lambda df: df.head(-val_periods)).reset_index(drop=True)

    return train, val

def get_prediction_panel(y_panel_df, h, freq):
    """Construct panel to use with
    predict method.
    """
    df = y_complete_train_df[['unique_id', 'ds']].groupby('unique_id').max().reset_index()

    predict_panel = []
    for idx, df in df.groupby('unique_id'):
        date = df['ds'].values.item()
        unique_id = df['unique_id'].values.item()

        date_range = pd.date_range(date, periods=4, freq='D')
        df_ds = pd.DataFrame.from_dict({'ds': date_range})
        df_ds['unique_id'] = unique_id
        predict_panel.append(df_ds[['unique_id', 'ds']])

    predict_panel = pd.concat(predict_panel)

    return predict_panel
