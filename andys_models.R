library(tidyverse)
library(ggplot2)
library(tidymodels)
library(lubridate)

set.seed(42)

# ====================
# Feature Engineering
# ====================

# load data
train_path <- '~/Documents/stats_101c_project/ucla-stats-101-c-2024-su-regression/train.csv'
train_csv <- read.csv(train_path)

# remove order_totals
train_csv <- train_csv %>% select(-order_totals)

# train-test-split
train_split <- initial_split(
  data = train_csv, 
  prop = 0.8,
  strata = 'log_total'
)
train <- train_split %>% training()
test  <- train_split %>% testing()

# 10-fold cross validation
train_folds <- vfold_cv(
  data = train,
  v = 10,
  strata = 'log_total'
)

# create original recipe
recipe_0 <- recipe(
  log_total ~ .,
  data = train
)

# create simple recipe
recipe_1 <- recipe(
  log_total ~ .,
  data = train
) %>%
  # remove correlated predictors
  step_corr(
    all_numeric(),
    threshold = 0.8
  ) %>%
  # normalizes numeric predictors
  step_normalize(all_numeric()) %>% 
  # creates dummy variables for categorical predictors
  step_dummy(all_nominal(), -all_outcomes())

# create recipe for xgboost
recipe_2 <- recipe(
  log_total ~ .,
  data = train
) %>%
  # creates dummy variables for categorical predictors
  step_dummy(all_nominal(), -all_outcomes())

# train the recipe on training set
#recipe_1_prep <- recipe_1 %>% 
#  prep(training = train)

# use trained recipe to transform train and test sets
#train_prep <- recipe_1_prep %>% 
#  bake(new_data = NULL)
#test_prep <- recipe_1_prep %>% 
#  bake(new_data = test)

# seasonal indicators

# demographic ratios

# interaction features

# ====================
# Random Forest (HW3)
# ====================

rf_model <- rand_forest(
  #mtry = tune(),
  trees = tune(),
  min_n = tune()
) %>% 
  set_engine('ranger') %>% 
  set_mode('regression')

rf_wkfl <- workflow() %>% 
  add_model(rf_model) %>% 
  add_recipe(recipe_0)

rf_grid <- grid_random(
  parameters(rf_model),
  size = 10
)

rf_tune <- rf_wkfl %>% 
  tune_grid(
    resamples = train_folds,
    grid = rf_grid
  )

rf_tune %>% 
  collect_metrics() %>%
  filter(.metric == "rmse") %>% 
  mutate(mse = mean^2, rmse = mean) %>% 
  select(trees, min_n, mse, rmse)

best_rf_model <- rf_tune %>% 
  select_best(metric = 'rmse')

final_rf_wkfl <- rf_wkfl %>% 
  finalize_workflow(best_rf_model)

rf_final_fit <- final_rf_wkfl %>% 
  last_fit(split = train_split)

rf_final_fit %>% 
  collect_metrics() %>%
  filter(.metric == "rmse") %>% 
  mutate(mse = .estimate^2, rmse = .estimate) %>% 
  select(mse, rmse)

# mse: 0.0140

# ====================
# Linear Regression
# ====================

lm_model <- linear_reg(
  penalty = tune(),
  mixture = tune()
) %>% 
  set_engine('glmnet') %>% 
  set_mode('regression')

lm_wkfl <- workflow() %>% 
  add_model(lm_model) %>% 
  add_recipe(recipe_1)

lm_grid <- grid_random(
  parameters(lm_model),
  size = 10
)

lm_tune <- lm_wkfl %>% 
  tune_grid(
    resamples = train_folds,
    grid = lm_grid
  )

lm_tune %>% 
  collect_metrics() %>%
  filter(.metric == "rmse") %>% 
  mutate(mse = mean^2, rmse = mean) %>% 
  select(penalty, mixture, mse, rmse)

best_lm_model <-lm_tune %>% 
  select_best(metric = 'rmse')

final_lm_wkfl <- lm_wkfl %>% 
  finalize_workflow(best_lm_model)

lm_final_fit <- final_lm_wkfl %>% 
  last_fit(split = train_split)

lm_final_fit %>% 
  collect_metrics() %>%
  filter(.metric == "rmse") %>% 
  mutate(mse = .estimate^2, rmse = .estimate) %>% 
  select(mse, rmse)

# mse: 0.0814

# ====================
# Multi-Layer Perceptron
# ====================

nn_model <- mlp(
  hidden_units = tune(),
  penalty = tune(),
  #dropout = tune(),
  #learn_rate = tune(),
  epochs = 20
) %>% 
  set_engine('nnet') %>% 
  set_mode('regression')

nn_wkfl <- workflow() %>% 
  add_model(nn_model) %>% 
  add_recipe(recipe_1)

nn_grid <- grid_random(
  parameters(nn_model),
  size = 10
)

nn_tune <- nn_wkfl %>% 
  tune_grid(
    resamples = train_folds,
    grid = nn_grid
  )

nn_tune %>% 
  collect_metrics() %>%
  filter(.metric == "rmse") %>% 
  mutate(mse = mean^2, rmse = mean) %>% 
  select(hidden_units, penalty, mse, rmse)

best_nn_model <- nn_tune %>% 
  select_best(metric = 'rmse')

final_nn_wkfl <- nn_wkfl %>% 
  finalize_workflow(best_nn_model)

nn_final_fit <- final_nn_wkfl %>% 
  last_fit(split = train_split)

nn_final_fit %>% 
  collect_metrics() %>%
  filter(.metric == "rmse") %>% 
  mutate(mse = .estimate^2, rmse = .estimate) %>% 
  select(mse, rmse)

# mse: 0.103

# ====================
# Gradient-Boosting Trees
# ====================

xgb_model <- boost_tree(
  trees = tune(),
  min_n = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  loss_reduction = tune(),
  sample_size = tune(),
  stop_iter = tune()
) %>% 
  set_engine('xgboost') %>% 
  set_mode('regression')

xgb_wkfl <- workflow() %>% 
  add_model(xgb_model) %>% 
  add_recipe(recipe_2)

xgb_grid <- grid_random(
  parameters(xgb_model),
  size = 10
)

xgb_tune <- xgb_wkfl %>% 
  tune_grid(
    resamples = train_folds,
    grid = xgb_grid
  )

xgb_tune %>% 
  collect_metrics() %>%
  filter(.metric == "rmse") %>% 
  mutate(mse = mean^2, rmse = mean) %>% 
  select(trees, min_n, tree_depth, learn_rate, loss_reduction, sample_size, stop_iter, mse, rmse)

best_xgb_model <- xgb_tune %>% 
  select_best(metric = 'rmse')

final_xgb_wkfl <- xgb_wkfl %>% 
  finalize_workflow(best_xgb_model)

xgb_final_fit <- final_xgb_wkfl %>% 
  last_fit(split = train_split)

xgb_final_fit %>% 
  collect_metrics() %>%
  filter(.metric == "rmse") %>% 
  mutate(mse = .estimate^2, rmse = .estimate) %>% 
  select(mse, rmse)

# mse: 0.0142

# ====================
# Multivariate Adaptive Regression Splines
# ====================

mars_model <- mars(
  num_terms = tune(),
  prod_degree = tune(),
  prune_method = tune()
) %>% 
  set_engine('earth') %>% 
  set_mode('regression')

mars_wkfl <- workflow() %>% 
  add_model(mars_model) %>% 
  add_recipe(recipe_2)

mars_grid <- grid_random(
  parameters(mars_model),
  size = 10
)

mars_tune <- mars_wkfl %>% 
  tune_grid(
    resamples = train_folds,
    grid = mars_grid
  )

mars_tune %>% 
  collect_metrics() %>%
  filter(.metric == "rmse") %>% 
  mutate(mse = mean^2, rmse = mean) %>% 
  select(num_terms, prod_degree, prune_method, mse, rmse)

best_mars_model <- mars_tune %>% 
  select_best(metric = 'rmse')

final_mars_wkfl <- mars_wkfl %>% 
  finalize_workflow(best_mars_model)

mars_final_fit <- final_mars_wkfl %>% 
  last_fit(split = train_split)

mars_final_fit %>% 
  collect_metrics() %>%
  filter(.metric == "rmse") %>% 
  mutate(mse = .estimate^2, rmse = .estimate) %>% 
  select(mse, rmse)

# mse: 0.0160

# ====================
# Test Data
# ====================

test_path <- '~/Documents/stats_101c_project/ucla-stats-101-c-2024-su-regression/test.csv'
TEST_FINAL <- read.csv(test_path)

predictions <- TEST_FINAL %>% 
  select(id) %>% 
  bind_cols(
    mars_final_fit %>% 
      extract_workflow() %>% 
      predict(new_data = TEST_FINAL)
  ) %>% 
  rename(log_total = .pred)

write.csv(
  predictions, 
  "preds/preds_xgb_andy_0724.csv",
  row.names = FALSE
)