library(tidyverse)
library(tidymodels)

train <- read_csv("train.csv")
train <- train %>% 
  select(-order_totals)

set.seed(150)
train_folds <- vfold_cv(train, v = 10, strata = log_total)

rf_model <- rand_forest(mode = "regression", trees = 1000) %>% 
  set_engine("ranger")

rf_workflow <- workflow() %>% 
  add_model(rf_model) %>% 
  add_formula(log_total ~ .)

rf_res <- rf_workflow %>% 
  fit_resamples(resamples = train_folds)


rf_metrics <- rf_res %>% 
  collect_metrics()


rf_workflow_fit <- rf_workflow %>% 
  fit(data = train)

test <- read_csv("test.csv")


predictions <- predict(rf_workflow_fit, test) %>% 
  bind_cols(test %>% select(id))

final_predictions <- predictions %>% 
  select(id, log_total = .pred)