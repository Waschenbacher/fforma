"""This model replaces the original MetaModels based on R models defined in r-models.py.

The original r-models.py has the following models:
- ARIMA
- ETS
- ThetaF
- Naive
- SeasonalNaive
- RandomWalk
- NNETAR (Neural network): https://robjhyndman.com/hyndsight/nnetar-prediction-intervals/
- TBATS (for multiple seasonal patterns using exponential smooting):
https://robjhyndman.com/papers/ComplexSeasonality.pdf
- STLM: https://pkg.robjhyndman.com/forecast/reference/forecast.stl.html#:~:text=stlm%20forecasts%20the%20seasonally%20adjusted,reseasonalizes%2C%20and%20returns%20the%20forecasts.
"""

#!/usr/bin/env python
# coding: utf-8

from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
import pandas as pd
from copy import deepcopy
from statsforecast import StatsForecast
# TODO: replace the utils with utilsforecast
from esrnn_Di.esrnn_utils_evaluation import smape, mase, evaluate_panel

def check_is_fitted(estimator: Any, attributes: Union[str, List[str], Tuple[str, ...]]) -> bool:
    """Check if a model is fitted.

    Parameters
    ----------
    estimator : Any
        Model instance to check
    attributes : Union[str, List[str], Tuple[str, ...]]
        Attribute name(s) to check for

    Returns
    -------
    bool
        True if model is fitted

    Raises
    ------
    TypeError
        If estimator doesn't have a fit method
    ValueError
        If model is not fitted (missing attributes)
    """
    if not hasattr(estimator, 'fit'):
        raise TypeError("%s is not an estimator instance." % estimator)

    if not isinstance(attributes, (list, tuple)):
        attributes = [attributes]

    if not all([hasattr(estimator, attr) for attr in attributes]):
        raise ValueError(f"This {type(estimator).__name__} instance is not fitted yet.")

    return True
# TODO: use infer_frey based on df_y_panel['ds'] to infer the frequency. Write an
#  additional util to convert frequency from pandas to statsforecast.
def get_freq_for_statsforecast(seasonality: int) -> str:
    if seasonality == 24:
        return "H"
    elif seasonality == 7:  # Weekly seasonality with daily frequency
        return "D"
    elif seasonality == 365 or seasonality == 30:
        return "D"
    elif seasonality == 12:
        return "M"
    elif seasonality == 52 or seasonality == 54:
        return "W"
    else:
        raise ValueError(f"Invalid seasonality: {seasonality}")


def create_statsforecast_column_mapping(statsforecast_models: List[Any], model_names: List[str]) -> Dict[str, str]:
    """
    Create column mapping from StatsForecast model instances to desired names.

    Parameters
    ----------
    statsforecast_models : List[Any]
        List of StatsForecast model instances
    model_names : List[str]
        List of desired model names

    Returns
    -------
    Dict[str, str]
        Mapping from expected StatsForecast column names to desired model names
    """
    column_mapping = {}

    for model_instance, model_name in zip(statsforecast_models, model_names):
        # StatsForecast typically uses the model class name as column name
        statsforecast_column = model_instance.__class__.__name__
        column_mapping[statsforecast_column] = model_name

    return column_mapping

class MetaModels:
    """
    Efficient ensemble model trainer using direct StatsForecast batch processing.

    This optimized version uses a single StatsForecast object for all statsforecast models,
    providing significant performance improvements over individual model fitting.

    Key Features:
    - Single StatsForecast object handles ALL statsforecast models simultaneously
    - Batch processing for all time series at once
    - Mixed model support (statsforecast + non-statsforecast models)
    - Automatic seasonality parameter injection

    Parameters
    ----------
    models: dict
        Dictionary of models to train. Can be:
        - StatsForecast model classes (AutoARIMA, AutoETS, etc.)
        - Non-StatsForecast model classes (Naive2, custom models)
        - Model instances (will be deepcopied)
    scheduler: str, optional
        Kept for compatibility but not used in this implementation
    seasonality: int, optional
        Default seasonality to use when instantiating models that require season_length
    """

    def __init__(self,
                 models: Dict[str, Union[type, Any]],
                 scheduler: str = 'processes',
                 seasonality: int = 7) -> None:
        self.models = models
        self.scheduler = scheduler  # Kept for compatibility
        self.seasonality = seasonality

    def _instantiate_model(self, model: Union[type, Any]) -> Any:
        """Create a model instance from either a class or existing instance.

        Parameters
        ----------
        model : Union[type, Any]
            Model class or instance

        Returns
        -------
        Any
            Instantiated model object
        """
        if isinstance(model, type):
            # It's a class, instantiate it with appropriate parameters
            try:
                # Try to instantiate without parameters first
                return model()
            except TypeError as e:
                error_str = str(e)
                if 'season_length' in error_str:
                    # Model requires season_length parameter (StatsForecast models)
                    return model(season_length=self.seasonality)
                elif 'seasonality' in error_str:
                    # Model requires seasonality parameter (ESRNN models like Naive2)
                    return model(seasonality=self.seasonality)
                else:
                    # Other parameter issues, re-raise
                    raise
        else:
            # It's already an instance, deepcopy it
            return deepcopy(model)

    def _fit_statsforecast(self, y_panel_df: pd.DataFrame, statsforecast_models: List[Any], statsforecast_model_names: List[str]) -> None:
        """
        Fit StatsForecast models using batch processing.

        Parameters
        ----------
        y_panel_df : pd.DataFrame
            Training data with columns ['unique_id', 'ds', 'y']
        statsforecast_models : List[Any]
            List of instantiated StatsForecast model objects
        statsforecast_model_names : List[str]
            List of model names corresponding to statsforecast_models
        """
        if not statsforecast_models:
            self.statsforecast_obj_ = None
            self.statsforecast_model_names_ = []
            return

        try:
            self.statsforecast_obj_ = StatsForecast(
                models=statsforecast_models,
                freq=get_freq_for_statsforecast(self.seasonality),
                n_jobs=1
            )
            self.statsforecast_model_names_ = statsforecast_model_names

            # Create column mapping beforehand during fit phase
            self.statsforecast_column_mapping_ = create_statsforecast_column_mapping(
                statsforecast_models, statsforecast_model_names
            )

            # Fit the StatsForecast object to store fitted models
            self.statsforecast_obj_.fit(y_panel_df)
            print(f"✅ Fitted {len(statsforecast_models)} StatsForecast models successfully")

        except Exception as e:
            raise RuntimeError(f"Failed to fit StatsForecast models: {e}")

    def _fit_non_statsforecast(self, y_panel_df: pd.DataFrame, non_statsforecast_models: Dict[str, Any]) -> None:
        """
        Fit non-StatsForecast models individually for each time series.

        Parameters
        ----------
        y_panel_df : pd.DataFrame
            Training data with columns ['unique_id', 'ds', 'y']
        non_statsforecast_models : Dict[str, Any]
            Dictionary of model_name -> model_instance for non-StatsForecast models
        """
        self.non_statsforecast_models_ = {}

        for model_name, model_template in non_statsforecast_models.items():
            self.non_statsforecast_models_[model_name] = {}

            for unique_id, group in y_panel_df.groupby('unique_id'):
                try:
                    model_instance = self._instantiate_model(self.models[model_name])
                    y = group['y'].values
                    if hasattr(model_instance, 'fit'):
                        model_instance.fit(y)
                    self.non_statsforecast_models_[model_name][unique_id] = model_instance
                except Exception as e:
                    raise RuntimeError(f"Warning: Failed to fit {model_name} for {unique_id}:"
                             f" {e}")

    def fit(self, y_panel_df: pd.DataFrame) -> 'MetaModels':
        """Fit models using efficient batch processing for StatsForecast models.

        Parameters
        ----------
        y_panel_df : pd.DataFrame
            DataFrame with columns ['unique_id', 'ds', 'y']

        Returns
        -------
        MetaModels
            Returns self for method chaining
        """
        # Validate input format
        required_cols = ['unique_id', 'ds', 'y']
        if not all(col in y_panel_df.columns for col in required_cols):
            raise ValueError(f"Input DataFrame must have columns {required_cols}")

        # Store the training data for use in prediction
        self.fitted_data_ = y_panel_df.copy()

        # Separate statsforecast models from non-statsforecast models
        statsforecast_models = []
        statsforecast_model_names = []
        non_statsforecast_models = {}

        for model_name, model in self.models.items():
            model_instance = self._instantiate_model(model)

            if hasattr(model_instance, '__module__') and 'statsforecast' in str(model_instance.__module__):
                statsforecast_models.append(model_instance)
                statsforecast_model_names.append(model_name)
            else:
                non_statsforecast_models[model_name] = model_instance

        # Fit models using utility methods for better organization
        self._fit_statsforecast(y_panel_df, statsforecast_models, statsforecast_model_names)
        self._fit_non_statsforecast(y_panel_df, non_statsforecast_models)

        return self

    @staticmethod
    def _combine_predictions(df_predictions: List[Optional[pd.DataFrame]]) -> pd.DataFrame:
        """Combine predictions to a DF with each column being predictions of a model."""
        # Remove None from df_predictions
        df_predictions_clean = [df for df in df_predictions if df is not None]

        if not df_predictions_clean:
            raise ValueError("No valid predictions found (all predictions are None)")
        if len(df_predictions_clean) == 1:
            return df_predictions_clean[0]

        # Merge all predictions on unique_id and ds
        df_combined = df_predictions_clean[0]
        for df in df_predictions_clean[1:]:
            df_combined = pd.merge(df_combined, df, on=['unique_id', 'ds'], how='outer')
        return df_combined


    def _predict_statsforecast(self, y_hat_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Generate predictions using fitted StatsForecast models.

        Parameters
        ----------
        y_hat_df : pd.DataFrame
            DataFrame with columns ['unique_id', 'ds'] specifying where to forecast

        Returns
        -------
        Optional[pd.DataFrame]
            List of prediction DataFrames for StatsForecast models, or None if no models
        """
        if hasattr(self, 'statsforecast_obj_') and self.statsforecast_obj_:
            try:
                # TODO: do we need h or can we use y_hat_df in .predict() directly?
                # Calculate forecast horizon for each time series
                unique_ids = y_hat_df['unique_id'].unique()
                h_dict = {}
                for uid in unique_ids:
                    h_dict[uid] = len(y_hat_df[y_hat_df['unique_id'] == uid])

                # Assume all series have same forecast horizon for simplicity
                h = max(h_dict.values()) if h_dict else 1

                # Use predict() method on already fitted StatsForecast object
                print(f"Generating predictions using fitted StatsForecast models (h={h})...")
                forecasts = self.statsforecast_obj_.predict(h=h)

                # Use pre-created column mapping (created during fit phase)
                if hasattr(self, 'statsforecast_column_mapping_'):
                    forecasts = forecasts.rename(columns=self.statsforecast_column_mapping_)
                    print(f"Applied column mapping: {self.statsforecast_column_mapping_}")
                else:
                    print("Warning: No pre-created column mapping found, using original column names")

                # Filter forecasts to match requested forecast dates
                forecasts = forecasts.merge(y_hat_df, on=['unique_id', 'ds'], how='inner')

                if forecasts.empty:
                    raise RuntimeError(f"Failed to generate predictions for {y_hat_df.shape[0]} rows")

                print(f"✅ Generated predictions using fitted models for {len(self.statsforecast_model_names_)} StatsForecast models")
                return forecasts

            except Exception as e:
                raise RuntimeError(f"Warning: Failed to generate StatsForecast predictions from "
                   f"fitted models: {e}")
        return None

    def _predict_non_statsforecast(self, y_hat_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Generate predictions using fitted non-StatsForecast models.

        Parameters
        ----------
        y_hat_df : pd.DataFrame
            DataFrame with columns ['unique_id', 'ds'] specifying where to forecast

        Returns
        -------
        Optional[pd.DataFrame]
            List of prediction DataFrames for non-StatsForecast models
        """
        predictions = []

        if hasattr(self, 'non_statsforecast_models_') and self.non_statsforecast_models_:
            for model_name, fitted_models_dict in self.non_statsforecast_models_.items():
                model_predictions = []

                for unique_id, group in y_hat_df.groupby('unique_id'):
                    if unique_id in fitted_models_dict and fitted_models_dict[unique_id] is not None:
                        h = len(group)
                        model_instance = fitted_models_dict[unique_id]

                        try:
                            # Use sklearn-style predict method
                            if hasattr(model_instance, 'predict'):
                                preds = model_instance.predict(h)
                            else:
                                preds = [0.0] * h

                            # Handle predictions format
                            if hasattr(preds, 'values'):
                                preds = preds.values
                            elif not isinstance(preds, (list, np.ndarray)):
                                preds = [preds] * h

                            # Ensure we have the right number of predictions
                            preds = np.array(preds).flatten()
                            if len(preds) < h:
                                preds = np.concatenate([preds, np.repeat(preds[-1] if len(preds) > 0 else 0.0, h - len(preds))])
                            elif len(preds) > h:
                                preds = preds[:h]

                            pred_df = pd.DataFrame({
                                'unique_id': unique_id,
                                'ds': group['ds'].values,
                                model_name: preds
                            })
                            model_predictions.append(pred_df)

                        except Exception as e:
                            raise RuntimeError(f"Warning: Failed to predict {model_name} for {unique_id}: {e}")
                    else:
                        # Series not fitted for this model
                        h = len(group)
                        pred_df = pd.DataFrame({
                            'unique_id': unique_id,
                            'ds': group['ds'].values,
                            model_name: [0.0] * h
                        })
                        model_predictions.append(pred_df)

                if model_predictions:
                    model_df = pd.concat(model_predictions, ignore_index=True)
                    predictions.append(model_df)
            return self._combine_predictions(predictions)
        else:
            return None

    def predict(self, y_hat_df: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions using efficient batch processing for StatsForecast models.

        Parameters
        ----------
        y_hat_df : pd.DataFrame
            DataFrame with columns ['unique_id', 'ds'] specifying where to forecast

        Returns
        -------
        pd.DataFrame
            DataFrame with forecasts in wide format (one column per model)
        """
        # Check if we have fitted models
        required_attrs = []
        if hasattr(self, 'statsforecast_obj_'):
            required_attrs.append('statsforecast_obj_')
        if hasattr(self, 'non_statsforecast_models_'):
            required_attrs.append('non_statsforecast_models_')

        if not required_attrs:
            raise ValueError("This MetaModels instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")

        # Validate input format
        required_cols = ['unique_id', 'ds']
        if not all(col in y_hat_df.columns for col in required_cols):
            raise ValueError(f"Input DataFrame must have columns {required_cols}")

        # Use utility methods to handle different model types
        statsforecast_predictions = self._predict_statsforecast(y_hat_df)
        other_predictions = self._predict_non_statsforecast(y_hat_df)

        # Combine all predictions
        return self._combine_predictions([statsforecast_predictions, other_predictions])


################################################################################
########## UTILS FOR FFORMA FLOW (unchanged)
###############################################################################

def temp_holdout(y_panel_df: pd.DataFrame, val_periods: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits the data in train and validation sets.

    Parameters
    ----------
    y_panel_df : pd.DataFrame
        Pandas DataFrame with columns ['unique_id', 'ds', 'y']
    val_periods : int
        Number of periods to hold out for validation

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Train and validation DataFrames
    """
    val = y_panel_df.groupby('unique_id').tail(val_periods)
    train = y_panel_df.groupby('unique_id').apply(lambda df: df.head(-val_periods)).reset_index(drop=True)

    return train, val


def calc_errors(y_panel_df: pd.DataFrame,
                 y_insample_df: pd.DataFrame,
                 seasonality: int,
                 benchmark_model: str = 'Naive2') -> pd.DataFrame:
    """Calculates OWA of each time series using benchmark_model as benchmark.

    Parameters
    ----------
    y_panel_df : pd.DataFrame
        Pandas DataFrame with columns ['unique_id', 'ds', 'y']
    y_insample_df : pd.DataFrame
        Pandas DataFrame with columns ['unique_id', 'ds', 'y'] (Train set)
    seasonality : int
        Frequency of the time series
    benchmark_model : str, default='Naive2'
        Column name of the benchmark model

    Returns
    -------
    pd.DataFrame
        OWA errors for each time series and each model
    """
    assert benchmark_model in y_panel_df.columns

    y_panel = y_panel_df[['unique_id', 'ds', 'y']]
    y_hat_panel_fun = lambda model_name: y_panel_df[['unique_id', 'ds', model_name]].rename(columns={model_name: 'y_hat'})

    model_names = set(y_panel_df.columns) - set(y_panel.columns)

    errors_smape = y_panel[['unique_id']].drop_duplicates().reset_index(drop=True)
    errors_mase = errors_smape.copy()

    for model_name in model_names:
        errors_smape[model_name] = None
        errors_mase[model_name] = None
        y_hat_panel = y_hat_panel_fun(model_name)

        errors_smape[model_name] = evaluate_panel(y_panel, y_hat_panel, smape)
        errors_mase[model_name] = evaluate_panel(y_panel, y_hat_panel, mase, y_insample_df, seasonality)

    mean_smape_benchmark = errors_smape[benchmark_model].mean()
    mean_mase_benchmark = errors_mase[benchmark_model].mean()

    errors_smape = errors_smape.drop(columns=benchmark_model).set_index('unique_id')
    errors_mase = errors_mase.drop(columns=benchmark_model).set_index('unique_id')

    errors = errors_smape/mean_mase_benchmark + errors_mase/mean_smape_benchmark
    errors = 0.5*errors

    return errors


def get_prediction_panel(y_panel_df: pd.DataFrame, h: int, freq: Optional[str]) -> pd.DataFrame:
    """Construct panel to use with predict method.

    Parameters
    ----------
    y_panel_df : pd.DataFrame
        Input panel data with columns ['unique_id', 'ds', 'y']
    h : int
        Number of periods to forecast
    freq : Optional[str]
        Frequency parameter (currently unused in implementation)

    Returns
    -------
    pd.DataFrame
        Prediction panel with future dates for each unique_id
    """
    df = y_panel_df[['unique_id', 'ds']].groupby('unique_id').max().reset_index()

    predict_panel = []
    for idx, row in df.iterrows():
        unique_id = row['unique_id']
        last_date = row['ds']

        # Generate future dates
        if hasattr(last_date, 'freq'):
            date_range = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=h, freq='D')
        else:
            date_range = pd.date_range(start=last_date, periods=h, freq='D')[1:]

        df_ds = pd.DataFrame({
            'unique_id': unique_id,
            'ds': date_range
        })
        predict_panel.append(df_ds)

    if predict_panel:
        predict_panel = pd.concat(predict_panel, ignore_index=True)
    else:
        predict_panel = pd.DataFrame(columns=['unique_id', 'ds'])

    return predict_panel