from utilsforecast.losses import mase, smape


def owa(y_hat_panel, y_panel, y_insample,
        seasonality, col_naive2: str = 'y_hat_naive2', id_col: str ='unique_id',
        date_col: str ='ds', target_col: str ='y'):
    """Calculates Overall Weighted Average (OWA) metric for forecasting models.

    Computes MASE and sMAPE for current models relative to Naive2 benchmark,
    then calculates the Overall Weighted Average as their mean ratio.

    Args:
        y_hat_panel (pd.DataFrame): Panel with forecast predictions and naive2 benchmark.
            Must contain columns: unique_id, ds, y_hat, y_hat_naive2.
        y_panel (pd.DataFrame): Panel with actual values.
            Must contain columns: unique_id, ds, y.
        y_insample (pd.DataFrame): Panel with training data for MASE calculation.
            Must contain columns: unique_id, ds, y.
        seasonality (int): Main frequency of the time series.
            Examples: Quarterly=4, Daily=7, Monthly=12.
        col_naive2 (str, optional): Column name for Naive2 predictions.
            Defaults to 'y_hat_naive2'.
        id_col (str, optional): Column name for unique identifiers.
            Defaults to 'unique_id'.
        date_col (str, optional): Column name for time periods.
            Defaults to 'ds'.
        target_col (str, optional): Column name for target values.
            Defaults to 'y'.

    Returns:
        tuple[pd.Series, pd.Series, pd.Series]: A tuple containing:
            - model_owa: OWA ratios for each model (excluding naive2)
            - model_mase: MASE values for each model (excluding naive2)
            - model_smape: sMAPE values for each model (excluding naive2)

    Raises:
        RuntimeError: If required columns are missing from input DataFrames.
    """
    # TODO: do we need to add preprocess step, e.g. sort etc.?
    # Validate required columns
    required_y_hat = {col_naive2, id_col, date_col}
    required_y_panel = {id_col, date_col, target_col}
    if not required_y_hat.issubset(y_hat_panel.columns):
        raise RuntimeError(f'y_hat_panel missing: {required_y_hat - set(y_hat_panel.columns)}')
    if not required_y_panel.issubset(y_panel.columns):
        raise RuntimeError(f'y_panel missing: {required_y_panel - set(y_panel.columns)}')

    # join y_panel and y_hat_panel on unique_id, ds
    y_combined = y_hat_panel.merge(y_panel, on=[id_col, date_col], how='inner')

    # Evaluate mase and smape
    models_to_evaluate = list(set(y_combined.columns) - set(y_insample.columns))
    total_mase = mase(df=y_combined, models=models_to_evaluate, train_df=y_insample,
                                                                        seasonality=seasonality)
    total_smape = smape(df=y_combined, models=models_to_evaluate)

    # Drop unique_id column and calculate mean for each model
    mase_means = total_mase.drop(columns=[id_col]).mean()
    smape_means = total_smape.drop(columns=[id_col]).mean() * 100

    model_mase = mase_means.drop(col_naive2)
    model_smape = smape_means.drop(col_naive2)

    # Extract naive2 benchmark values
    naive2_mase = mase_means[col_naive2]
    naive2_smape = smape_means[col_naive2]

    # Calculate owa for all models (divide each column by col_naive2)
    model_owa = ((model_mase / naive2_mase + model_smape / naive2_smape) / 2)

    return model_owa, model_mase, model_smape
