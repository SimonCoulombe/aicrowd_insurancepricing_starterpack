library(tidyverse)
library(tidymodels)

training <- read_csv("training_data.csv")

my_first_recipe <-
  recipes::recipe(
    claim_amount ~ .,
    training[0, ]
  ) %>%
  recipes::step_mutate(
    age_when_licensed = drv_age1 - drv_age_lic1,
  ) %>%
  recipes::step_string2factor(recipes::all_nominal()) %>%
  recipes::step_other(recipes::all_nominal(), threshold = 0.05) %>% ##  categories with less than 5% of total are grouped in the "other" group
  recipes::step_dummy(all_nominal(), one_hot = TRUE) %>% # one-hot encode categories
  step_rm(contains("id_policy")) # remove ID

prepped_first_recipe <- recipes::prep(my_first_recipe, training, retain = FALSE)

write_rds(prepped_first_recipe, "prepped_first_recipe.rds")


baked_data_first_recipe <- recipes::bake(prepped_first_recipe, new_data = training)
feature_names <- baked_data_first_recipe %>%
  select(-claim_amount) %>%
  colnames()
write_rds(feature_names, "feature_names.rds")
