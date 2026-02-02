#!/usr/bin/env python
# coding: utf-8
import pandas as pd

from fforma import FFORMA
from statsforecast.models import (
    AutoETS,
    OptimizedTheta,
    Naive as StatsForecastNaive,
    SeasonalNaive as StatsForecastSeasonalNaive,
)

from fforma.meta_model_statsforecast import (
    MetaModels,
    temp_holdout,
    calc_errors,
)
from esrnn_Di.esrnn_m4_data import prepare_m4_data, seas_dict
from esrnn_Di.esrnn_utils_evaluation import Naive2
from tsfeatures import tsfeatures


def prepare_to_train_fforma(dataset, validation_periods, seasonality):

    X_train_df, y_train_df, X_test_df, y_test_df = prepare_m4_data(
        dataset, "./R/data", 100
    )

    # There must be a good amount of time series. Otherwise FForma will fail,
    # as lightgbm needs enough samples when there are many features.
    dev_series = [f"W{i}" for i in range(1, 100)]  # [W1, W2, ..., W20] - More series
    # for meaningful features
    y_train_df = y_train_df[y_train_df["unique_id"].isin(dev_series)]
    y_test_df = y_test_df[y_test_df["unique_id"].isin(dev_series)]

    # Preparing errors
    y_holdout_train_df, y_val_df = temp_holdout(y_train_df, validation_periods)
    meta_models = {
        "ETS": AutoETS,
        "ThetaF": OptimizedTheta,
        "Naive": StatsForecastNaive,
        "SeasonalNaive": StatsForecastSeasonalNaive,
        "Naive2": Naive2(seasonality=2),
    }
    validation_meta_models = MetaModels(meta_models, seasonality=seasonality)
    validation_meta_models.fit(y_holdout_train_df)
    prediction_validation_meta_models = validation_meta_models.predict(y_val_df)

    # Calculating errors
    errors = calc_errors(
        prediction_validation_meta_models, y_holdout_train_df, seasonality
    )

    # Calculating features
    features = tsfeatures(y_holdout_train_df, seasonality)

    # Calculating actual predictions
    final_meta_models = MetaModels(meta_models, seasonality=seasonality)
    final_meta_models.fit(y_train_df)

    predictions = final_meta_models.predict(y_test_df[["unique_id", "ds"]])

    return errors, features, predictions


def main():
    complete_errors, complete_features, complete_predictions = [], [], []

    for dataset in ["Weekly"]:  #'Daily', etc
        validation_periods = seas_dict[dataset]["output_size"]
        seasonality = seas_dict[dataset]["seasonality"]
        errors, features, predictions = prepare_to_train_fforma(
            dataset, validation_periods, seasonality
        )

        complete_errors.append(errors)
        complete_features.append(features)
        complete_predictions.append(predictions)

    complete_errors = pd.concat(complete_errors)
    complete_features = pd.concat(complete_features)
    complete_predictions = pd.concat(complete_predictions)

    # Training fforma

    # optimal params by hyndman
    optimal_params = {
        "n_estimators": 94,
        "eta": 0.58,
        "max_depth": 14,
        "subsample": 0.92,
        "colsample_bytree": 0.77,
    }
    fforma = FFORMA(params=optimal_params)
    fforma.fit(
        errors=complete_errors, holdout_feats=complete_features, feats=complete_features
    )

    fforma_predictions = fforma.predict(complete_predictions)
    print(fforma_predictions)

    # evaluate predictions


if __name__ == "__main__":
    main()
