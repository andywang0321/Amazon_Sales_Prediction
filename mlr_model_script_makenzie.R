library(tidyverse)
library(rsample)
library(tidymodels)

training <- read_csv("train.csv")
testing_set <- read_csv("test.csv")

training_set <- training %>%
  select(-order_totals)

set.seed(16)

# cross-validation
rep_training_folds <- vfold_cv(training_set, v = 10, repeats = 5, strata = 'log_total')

# simple linear regression
lm_model <- linear_reg() %>%
  set_engine("lm")

lm_training_fit <- lm_model %>%
  fit(log_total ~ count, data = training_set)

lm_predictions <- training_set %>%
  select(count, log_total) %>%
  bind_cols(predict(lm_training_fit, new_data = training_set))

# multiple linear regression workflow

lm_workflow <- workflow() %>%
  add_model(lm_model)

regression_list <- list(
  year = log_total ~ year,
  month = log_total ~ month,
  count = log_total ~ count,
  female = log_total ~ count_female,
  male = log_total ~ count_male,
  order_less = log_total ~ count_less5,
  order_moderate = log_total ~ count_5to10,
  order_more = log_total ~ count_over10,
  combined_all = log_total ~ year + month + count + count_female + count_male + count_less5 + count_5to10 + count_over10,
  combined_order_freq = log_total ~  year + month + count + count_less5 + count_5to10 + count_over10,
  combined_gender = log_total ~ year + month + count + count_female + count_male,
  all_preds = log_total ~ .
)

training_models <- workflow_set(preproc = regression_list,
                                models = list(lm = lm_model))
training_models_fit <- training_models %>%
  workflow_map("fit_resamples", resamples = rep_training_folds)

mlr_predictions <- predict(rf_workflow_fit, new_data = testing_set)

mlr_pred <- bind_cols(testing_set%>% select(id), mlr_predictions)
mlr_pred <- mlr_pred %>%
  rename(log_total = .pred)

write_csv(mlr_pred, "preds_mlr_model_makenzie.csv")

