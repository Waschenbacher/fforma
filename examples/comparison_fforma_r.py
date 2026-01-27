import pandas as pd
from esrnn_Di.esrnn_m4_data import prepare_m4_data
from metrics.metrics import owa
from fforma import FFORMA
from functools import partial
import glob
import logging

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

freqs = {'Hourly': 24, 'Daily': 1,
         'Monthly': 12, 'Quarterly': 4,
         'Weekly':1, 'Yearly': 1
         }

train_errors = pd.read_csv(
    '../R/data/train-errors-fforma.csv').set_index('unique_id').sort_index()
train_feats = pd.read_csv(
    '../R/data/train-feats-fforma.csv').set_index('unique_id').sort_index()

feats = pd.read_csv(
    '../R/data/pred-feats-fforma.csv').set_index('unique_id').sort_index()
preds = [pd.read_csv(file) for file in glob.glob('../R/data/preds-fforma-*.csv')]
preds = pd.concat(preds).set_index(['unique_id', 'ds']).sort_index()


optimal_params = {'n_estimators': 94,
                  'eta': 0.58,
                  'max_depth': 14,
                  'subsample': 0.92,
                  'colsample_bytree': 0.77}
model = FFORMA(params=optimal_params, verbose_eval=20)

model.fit(errors=train_errors, holdout_feats=train_feats, feats=feats)

fforma_preds = model.predict(preds).reset_index()

fforma_preds.loc[fforma_preds['fforma_prediction']<0, 'fforma_prediction'] = 0


def evaluate_fforma(dataset_name, y_hat_df, directory, num_obs):
    print(dataset_name)

    # Target of test data is y_test_df; target of train data is y_train_df
    _, y_train_df, _, y_test_df = prepare_m4_data(dataset_name=dataset_name,
                                                  directory=directory,
                                                  num_obs=num_obs)

    # Convert column 'ds' with integer starting with 1 and with step 1
    y_test_df['ds'] = y_test_df.groupby('unique_id').cumcount() + 1

    # Keep forecast of interest
    y_hat_df = fforma_preds[
        y_hat_df['unique_id'].isin(y_test_df['unique_id'].unique())]

    # sort both y_train_df and y_test_df on unique_id and ds
    y_train_df = y_train_df.sort_values(['unique_id', 'ds'])
    y_test_df = y_test_df.sort_values(['unique_id', 'ds'])

    # Append forecast of Naive2 to y_hat_panel: change column ds for compatibility
    y_naive2_panel = y_test_df.filter(['unique_id', 'ds', 'y_hat_naive2'])

    y_hat_panel = y_hat_df.merge(y_naive2_panel,
                                 on=['unique_id', 'ds'], how='inner')

    seasonality = freqs[dataset_name]
    metric_owa, mase, smape = owa(y_hat_panel,
                                  y_panel=y_test_df.filter(['unique_id', 'ds', 'y']),
                                  y_insample=y_train_df.filter(
                                      ['unique_id', 'ds', 'y']),
                                  seasonality=seasonality)

    logger.info('=' * 15 + ' Model evaluation ' + '=' * 14)
    logger.info('OWA:\n{}'.format(metric_owa.round(3)))
    logger.info('SMAPE:\n{}'.format(smape.round(3)))
    logger.info('MASE:\n{}'.format(mase.round(3)))


    return dataset_name, metric_owa, mase, smape

evaluate_fforma_p = partial(evaluate_fforma, y_hat_df=fforma_preds, directory='.'
                                                                            './R/data', num_obs=100000)
eval_fforma = [evaluate_fforma_p(freq) for freq in freqs.keys()]

eval_fforma

