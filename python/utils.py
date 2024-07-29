import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

def get_year(date: str) -> int:
    assert len(date) == 10, "Date must be str of format 'yyyy-mm-dd'!"
    return int(date[0:4])

def get_month(date: str) -> int:
    assert len(date) == 10, "Date must be str of format 'yyyy-mm-dd'!"
    return int(date[5:7])

def mc_aggregate(
    amazon: pd.DataFrame, 
    customers: pd.DataFrame, 
    train: bool = False
) -> pd.DataFrame:
    '''
    Data transformation pipeline identical to the one Miles Chen used 
    to combine order_details and customers into train.csv

    amazon: pd.DataFrame with the following columns:
        - shipping_address_state
        - order_date
        - survey_response_id
        - item_cost
    '''
    orders = amazon[
        ~amazon.shipping_address_state.isnull()
    ].groupby(
        ['order_date', 'survey_response_id'], 
        as_index=False
    ).agg(
        total_cost = ('item_cost', 'sum')
    ).set_index(
        'survey_response_id'
    ).join(
        customers.set_index('survey_response_id'), 
        how='inner'
    ).reset_index()

    orders['year'] = orders.order_date.apply(get_year)
    orders['month'] = orders.order_date.apply(get_month)

    orders = orders.groupby(
        ['q_demos_state', 'year', 'month'], 
        as_index=False
    ).agg(
        order_totals = ('total_cost', 'sum')
    )

    orders['log_total'] = orders.order_totals.apply(np.log10)
    
    if train:
        orders = orders[orders.log_total > 2]

    return(orders)

class CustomDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.float32]:
        return self.X[idx], self.y[idx]

def join_and_prep(
    amazon: pd.DataFrame, 
    customers: pd.DataFrame
) -> pd.DataFrame:
    '''
    Joins amazon_order_details and customer_info and adds necessary columns
    to prepare for fine-grained modelling.

    returns:
        orders: pd.DataFrame with target variable total_cost.

    note that the target variable is total_cost, NOT log_total.
    '''
    orders = amazon[
        ~amazon.shipping_address_state.isnull()
    ].groupby(
        ['order_date', 'survey_response_id'], 
        as_index=False
    )
    
    if 'item_cost' in amazon.columns:
        orders = orders.agg(
            total_cost = ('item_cost', 'sum'),
            item_count = ('quantity', 'count')
        )
    else:
        orders = orders.agg(
            item_count = ('quantity', 'count')
        )
    
    orders = orders.set_index(
        'survey_response_id'
    ).join(
        customers.set_index('survey_response_id'), 
        how='inner'
    ).reset_index()

    orders['year'] = orders.order_date.apply(get_year)
    orders['month'] = orders.order_date.apply(get_month)

    orders = orders.drop(['order_date'], axis=1)
    orders = orders.drop(['q_life_changes'], axis=1)

    return orders

def aggregate_finegrained_preds(
    preds: torch.Tensor,
    _X: pd.DataFrame,
) -> pd.DataFrame:
    '''
    Aggregates fine-grained predictions into train.csv-style aggregated table.

    arguments:
        preds: torch.Tensor of model predictions (total_cost, NOT log_total)
        _X: pd.DataFrame of model features (prior to feature engineering), 
            must contain the following columns:
                - q_demos_state
                - year
                - month
    
    returns:
        _agg: pd.DataFrame with following columns:
                - q_demos_state
                - year
                - month
                - log_total
    
    IMPORTANT: _agg might contain more rows than test dataset.
               This is because rows in test.csv where log_total <= 2
               have been manually removed by MC. Since we do not have targets
               for test.csv, we would never know which rows were removed.

               Thus, to make predictions on test.csv, one more step is required.
    '''
    _X['total_cost'] = preds
    _agg = _X[
        ['q_demos_state', 'year', 'month', 'total_cost']
    ].groupby(
        ['q_demos_state', 'year', 'month'], 
        as_index=False
    ).agg(
        order_totals = ('total_cost', 'sum')
    )

    _agg['log_total'] = _agg.order_totals.apply(np.log10)
    _agg = _agg.drop(['order_totals'], axis=1)

    return _agg

def drop_unwanted_rows(my_df: pd.DataFrame, reference_df: pd.DataFrame) -> pd.DataFrame:
    '''
    Removes rows in my_df that is not present in reference_df.
        - reference_df must be subset of my_df. 
        - Both dfs must conetain columns: state, year, month (different colnames is fine)
    '''
    reference_set = set()

    for id, (state, year, month) in reference_df[['q_demos_state', 'year', 'month']].iterrows():
        row = (state, year, month)
        reference_set.add(row)

    for id, (state, year, month, preds) in my_df.iterrows():
        row = (state, year, month)
        if row not in reference_set:
            my_df = my_df.drop(id)

    return my_df.reset_index(drop=True)

def get_submission_preds(
    _agg: pd.DataFrame,
    test_csv: pd.DataFrame,
    save_path = None
) -> pd.DataFrame:
    '''
    Generates submission file by selecting rows of _agg present in test.csv,
    re-indexing using test.csv's index, and dropping all rows except id and log_total.

    If save_path is provided, a csv file will be saved.

    arguments:
        _agg: pd.DataFrame of aggregated fine-grained predictions.
            Use aggregate_finegrained_preds() to generate this.
        test_csv: pd.DataFrame of test.csv.
        [Optional] save_path: str of pathname to save destination. 
            If not provided, no save is made.
    
    returns:
        out: pd.DataFrame with columns id and log_total, appropriate for submission.
    '''
    test_ids = test_csv.id

    out = drop_unwanted_rows(
        _agg,
        test_csv[['q_demos_state', 'year', 'month']]
    )

    out['id'] = test_ids

    out = out.fillna(out.log_total.mean())

    if save_path is not None:
        out[['id', 'log_total']].to_csv(
            save_path, index=False
        )

    return out[['id', 'log_total']]