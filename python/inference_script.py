import pandas as pd
import numpy as np

from pickle import load
import torch

from utils import (
    # join amazon and customers
    join_and_prep,   
    # once predicted, aggregate predictions
    aggregate_finegrained_preds, 
    # attach correct ids in test.csv, remove rows not in test.csv, etc.
    get_submission_preds
)
# model definition
from model import FCNet


# load model

with open('final_pipeline.pkl', 'rb') as f:
    pipeline = load(f)

model = FCNet(148, [90], 0.1, 0.5)
model.load_state_dict(torch.load('final_model.pt', weights_only=True))


# load data

# Load customers_info_test
customers_test_path = "customer_info_test.csv"
customers_test = pd.read_csv(customers_test_path)
# Load order_details_test
amazon_test_path = "amazon_order_details_test.csv"
amazon_test = pd.read_csv(amazon_test_path)

# join them to create orders_test
orders_test = join_and_prep(amazon_test, customers_test)
# drop non-feature columns
_TEST = orders_test.drop(['survey_response_id'], axis=1)
# transform using pipeline
_TEST = pipeline.transform(_TEST)

# Load test.csv for submission indices
test_csv_path = "test.csv"
test_csv = pd.read_csv(test_csv_path)


# make preds

with torch.no_grad():
    model.cpu()
    preds = model(torch.from_numpy(_TEST.toarray()).to(torch.float32)).squeeze()

_agg = aggregate_finegrained_preds(preds, orders_test)


# save preds

save_path = f"submission.csv".replace(' ', '_')

get_submission_preds(
    _agg,
    test_csv,
    save_path
)