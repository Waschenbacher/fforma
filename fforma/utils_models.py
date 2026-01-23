from typing import Dict, Optional, Tuple, Union, Callable, Any
import pandas as pd
import lightgbm as lgb
import numpy as np

import copy
from packaging import version

from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# LightGBM version compatibility
LIGHTGBM_VERSION = version.parse(lgb.__version__)
LIGHTGBM_4X_CUSTOM_OBJECTIVE_BROKEN = LIGHTGBM_VERSION >= version.parse("4.0.0")

def _train_lightgbm(holdout_feats: Union[pd.DataFrame, np.ndarray],
                    best_models: np.ndarray,
                    params: Dict[str, Any],
                    fobj: Optional[Callable],
                    feval: Optional[Callable],
                    early_stopping_rounds: Optional[int],
                    verbose_eval: Union[bool, int],
                    seed: int) -> lgb.Booster:

    holdout_feats_train, holdout_feats_val, \
        best_models_train, \
        best_models_val, \
        indices_train, \
        indices_val = train_test_split(holdout_feats,
                                       best_models,
                                       np.arange(holdout_feats.shape[0]),
                                       random_state=seed,
                                       stratify=best_models
                                       )

    params = copy.deepcopy(params)
    num_round = int(params.pop('n_estimators', 100))

    params['num_class'] = len(np.unique(best_models))

    # LightGBM 4.x: Build callbacks list
    callbacks = []
    if early_stopping_rounds is not None and early_stopping_rounds > 0:
        callbacks.append(lgb.early_stopping(early_stopping_rounds))
    if verbose_eval is False:
        callbacks.append(lgb.log_evaluation(0))  # No logging
    elif verbose_eval is True:
        callbacks.append(lgb.log_evaluation(1))  # Log every iteration

    if fobj is not None:

        dtrain = lgb.Dataset(data=holdout_feats_train, label=indices_train)
        dvalid = lgb.Dataset(data=holdout_feats_val, label=indices_val)
        valid_sets = [dtrain, dvalid]

        if LIGHTGBM_VERSION >= version.parse("4.0.0"):
            # LightGBM 4.x: Custom objective in params (has bugs)
            params['objective'] = fobj
            gbm_model = lgb.train(
                params=params,
                train_set=dtrain,
                num_boost_round=num_round,
                feval=feval,
                valid_sets=valid_sets,
                callbacks=callbacks
            )
        else:
            # LightGBM 3.x: Custom objective as fobj parameter (correct approach)
            # Do NOT put custom objective in params - causes TypeError
            gbm_model = lgb.train(
                params=params,
                train_set=dtrain,
                num_boost_round=num_round,
                fobj=fobj,  # Custom objective as separate parameter
                feval=feval,
                valid_sets=valid_sets,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=verbose_eval
            )
    else:
        # Ensure default multiclass objective when no custom objective
        if 'objective' not in params:
            params['objective'] = 'multiclass'

        dtrain = lgb.Dataset(data=holdout_feats_train, label=best_models_train)
        dvalid = lgb.Dataset(data=holdout_feats_val, label=best_models_val)
        valid_sets = [dtrain, dvalid]

        if LIGHTGBM_VERSION >= version.parse("4.0.0"):
            # LightGBM 4.x: Use callbacks
            gbm_model = lgb.train(
                params=params,
                train_set=dtrain,
                num_boost_round=num_round,
                valid_sets=valid_sets,
                callbacks=callbacks
            )
        else:
            # LightGBM 3.x: Use old parameters
            gbm_model = lgb.train(
                params=params,
                train_set=dtrain,
                num_boost_round=num_round,
                valid_sets=valid_sets,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=verbose_eval
            )


    return gbm_model

def _train_lightgbm_cv(holdout_feats: Union[pd.DataFrame, np.ndarray],
                       best_models: np.ndarray,
                       params: Dict[str, Any],
                       fobj: Optional[Callable],
                       feval: Optional[Callable],
                       early_stopping_rounds: Optional[int],
                       verbose_eval: Union[bool, int],
                       seed: int,
                       folds: Callable,
                       train_model: bool = True) -> Union[lgb.Booster, Tuple[int, float]]:

    params = copy.deepcopy(params)
    num_round = int(params.pop('n_estimators', 100))

    params['num_class'] = len(np.unique(best_models))

    # LightGBM 4.x: Build callbacks list for CV
    callbacks = []
    if verbose_eval is False:
        callbacks.append(lgb.log_evaluation(0))  # No logging
    elif verbose_eval is True:
        callbacks.append(lgb.log_evaluation(1))  # Log every iteration

    if fobj is not None:
        print(f"🔧 Using custom objective in CV with LightGBM {LIGHTGBM_VERSION}")

        # Use indices as labels for custom objective (FFORMA requirement)
        indices = np.arange(holdout_feats.shape[0])
        dtrain = lgb.Dataset(data=holdout_feats, label=indices)

        if LIGHTGBM_VERSION >= version.parse("4.0.0"):
            # LightGBM 4.x: Custom objective in params (has bugs)
            params['objective'] = fobj
            gbm_model = lgb.cv(
                params=params,
                train_set=dtrain,
                num_boost_round=num_round,
                feval=feval,
                folds=folds(holdout_feats, best_models),
                callbacks=callbacks,
                seed=seed
            )
        else:
            # LightGBM 3.x: Custom objective as fobj parameter
            gbm_model = lgb.cv(
                params=params,
                train_set=dtrain,
                num_boost_round=num_round,
                fobj=fobj,  # Custom objective as separate parameter
                feval=feval,
                folds=folds(holdout_feats, best_models),
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=verbose_eval,
                seed=seed
            )
    else:
        # Ensure default multiclass objective when no custom objective
        if 'objective' not in params:
            params['objective'] = 'multiclass'

        dtrain = lgb.Dataset(data=holdout_feats, label=best_models)

        if LIGHTGBM_VERSION >= version.parse("4.0.0"):
            # LightGBM 4.x: Use callbacks
            gbm_model = lgb.cv(
                params=params,
                train_set=dtrain,
                num_boost_round=num_round,
                folds=folds(holdout_feats, best_models),
                callbacks=callbacks,
                seed=seed
            )
        else:
            # LightGBM 3.x: Use old parameters
            gbm_model = lgb.cv(
                params=params,
                train_set=dtrain,
                num_boost_round=num_round,
                folds=folds(holdout_feats, best_models),
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=verbose_eval,
                seed=seed
            )

    optimal_rounds = len(gbm_model[list(gbm_model.keys())[0]])
    best_performance = gbm_model[list(gbm_model.keys())[0]][-1]

    if train_model:
        params['n_estimators'] = optimal_rounds

        optimal_gbm_model = _train_lightgbm(holdout_feats, best_models,
                                            params, fobj, feval,
                                            early_stopping_rounds,
                                            verbose_eval, seed)

        return optimal_gbm_model

    return optimal_rounds, best_performance

def _train_lightgbm_grid_search(holdout_feats: Union[pd.DataFrame, np.ndarray],
                                best_models: np.ndarray,
                                use_cv: bool,
                                init_params: Dict[str, Any],
                                param_grid: Dict[str, Any],
                                fobj: Optional[Callable],
                                feval: Optional[Callable],
                                early_stopping_rounds: Optional[int],
                                verbose_eval: Union[bool, int],
                                seed: int,
                                folds: Callable) -> lgb.Booster:

    best_params = {}
    best_performance = np.inf

    pbar = tqdm(ParameterGrid(param_grid))
    pbar.set_description('Best performance: ??')

    for params in pbar:

        params = {**params, **init_params}

        if use_cv:
            num_round, performance =  _train_lightgbm_cv(holdout_feats, best_models,
                                                         params, fobj, feval,
                                                         early_stopping_rounds,
                                                         False, seed,
                                                         folds, train_model=False)
        else:
            gbm_model = _train_lightgbm(holdout_feats, best_models,
                                        params, fobj, feval,
                                        early_stopping_rounds,
                                        False, seed)
            performance = list(gbm_model.best_score['valid_1'].values())[0]
            num_round = gbm_model.best_iteration

        if performance < best_performance:
            #Updating  best performance
            pbar.set_description('Best performance: {}'.format(performance))
            #Updating bars
            best_params = params
            best_performance = performance
            best_params['n_estimators'] = num_round


    optimal_gbm_model = _train_lightgbm(holdout_feats, best_models,
                                        best_params, fobj, feval,
                                        early_stopping_rounds,
                                        False, seed)

    return optimal_gbm_model
