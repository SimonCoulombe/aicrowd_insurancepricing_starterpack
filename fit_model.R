fit_model <- function(x_raw= NULL, y_raw = NULL) {
  # Model training function: given training data (X_raw, y_raw), train this pricing model.

  # Parameters
  # ----------
  # X_raw : Dataframe, with the columns described in the data dictionary.
  # 	Each row is a different contract. This data has not been processed.
  # y_raw : a array, with the value of the claims, in the same order as contracts in X_raw.
  # 	A one dimensional array, with values either 0 (most entries) or >0.

  # Returns
  # -------
  # self: (optional), this instance of the fitted model.


  set.seed(42)
  source("preprocess_X_data.R")
  library(tidyverse)
  library(tidymodels)
  library(xgboost)

  
  if(is.null(x_raw)){
    training <- read_csv("training_data.csv")
  } else {
    training <- bind_cols(x_raw, y_raw)
  } 

  prepped_data <- preprocess_X_data(training)

  feature_names <- read_rds("feature_names.rds")

  xgtrain <- xgb.DMatrix(
    as.matrix(prepped_data %>% select(all_of(feature_names))),
    label = prepped_data %>% pull(claim_amount)
  )

  # create set of params to test
  my_params <- list(
    booster = "gbtree",
    objective = "reg:tweedie",
    eval_metric = "rmse",
    tweedie_variance_power = 1.5,
    gamma = 0,
    max_depth = 4,
    eta = 0.1,
    min_child_weight = 5,
    subsample = 0.6,
    colsample_bytree = 0.6,
    tree_method = "hist"
  )

  xgcv <- xgb.cv(
    params = my_params,
    data = xgtrain,
    nround = 200,
    nfold = 5,
    showsd = TRUE,
    early_stopping_rounds = 10
  )

  write_csv(tibble(best_test_rmse_mean = xgcv$evaluation_log$test_rmse_mean[xgcv$best_iteration]), "best_test_rmse_mean.csv")

  xgmodel <- xgboost::xgb.train(
    data = xgtrain,
    params = my_params,
    nrounds = xgcv$best_iteration
  )

  xgb.save(xgmodel, fname = "trained_model.xgb")

  my_importance <- xgb.importance(
    feature_names = feature_names,
    model = xgmodel
  ) %>%
    as_tibble()

  write_csv(my_importance, "my_importance.csv")

  # calculate adjustment so that we submit an illegal model that loses money on the training set
  preds <- predict(xgmodel, xgtrain)
  claims <- training$claim_amount

  ajusts <- sum(claims) / sum(preds)
  write_rds(ajusts, "ajusts.rds")

  return(xgmodel) # return(trained_model)
}
