library(tidyverse)
library(tidymodels)

train <- read_csv("train.csv")
train_new <- train %>%
  select(-order_totals)

set.seed(16)
train_fold <- vfold_cv(train_new, v = 10, strata = 'log_total')
rf_model <- rand_forest() %>%
  set_engine("ranger") %>%
  set_mode("regression")

rf_workflow <- workflow() %>%
  add_formula(log_total ~ .) %>%
  add_model(rf_model)

rf_resample <- control_resamples(save_pred = TRUE, save_workflow = TRUE)

rf_resample_fit <- rf_workflow %>%
  fit_resamples(resamples = train_fold, control = rf_resample)

rf_workflow_fit <- rf_workflow %>%
  fit(data = train_new)

test <- read_csv("test.csv")
rf_test_results <- predict(rf_workflow_fit, 
                           new_data = test)

rf_test_results <- bind_cols(test%>% select(id), rf_test_results)
rf_test_results <- rf_test_results %>%
  rename(log_total = .pred)

rf_test_results