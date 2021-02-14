library(tidyverse)
library(tidymodels)
library(xgboost)

#source("prep_recipe.R") # train wrangling recipes, only do onces
#source("fit_model.R")  # train xgboost, only do once.
#fit_model()
source("load_model.R")
source("predict_expected_claim.R")
source("predict_premium.R")
source("preprocess_X_data.R")

model_output_path <- "trained_model.xgb"
feature_names <- read_rds("feature_names.rds")
ajustement <- read_rds("ajusts.rds")
# This script expects sys.args arguments for (1) the dataset and (2) the output file.
output_dir = Sys.getenv('OUTPUTS_DIR', '.')
input_dataset = Sys.getenv('DATASET_PATH', 'training_data.csv')  # The default value.
output_claims_file = paste(output_dir, 'claims.csv', sep = '/')  # The file where the expected claims should be saved.
output_prices_file = paste(output_dir, 'prices.csv', sep = '/')  # The file where the prices should be saved.
model_output_path = 'trained_model.xgb'

feature_names <- read_rds("feature_names.rds")
ajustement <- read_rds("ajusts.rds")
if(!(interactive())){
  args = commandArgs(trailingOnly=TRUE)
  
  if(length(args) >= 1){
    input_dataset = args[1]
  }
  if(length(args) >= 2){
    output_claims_file = args[2]
  }
  if(length(args) >= 3){
    output_prices_file = args[3]
  }
} else message("not interactive")

Xraw <- read_csv(input_dataset) # load the data
x_clean <- preprocess_X_data(Xraw) # clean the data
trained_model <- load_model(model_output_path) # load the model

if (Sys.getenv("WEEKLY_EVALUATION", "false") == "true") {
  prices <- predict_premium(trained_model, x_clean)
  write.table(x = prices, file = output_prices_file, row.names = FALSE, col.names = FALSE, sep = ",")
} else {
  claims <- predict_expected_claim(trained_model, x_clean)
  write.table(x = claims, file = output_claims_file, row.names = FALSE, col.names = FALSE, sep = ",")
}
