# aicrowd_insurancepricing_starterpack

A starterpack for the insurance pricing competition at https://www.aicrowd.com/challenges/insurance-pricing-game

It uses the {recipes} package to prepare the data, ensuring that you will always have the same features at the end.

I used a "tweedie" model, might be new for some folks.  It allows you to model claim amount directly and  is an alternative to using a frequency + a severity model 

I also used the "recipes" package, which insure that  you won't create extra dummy variables by mistake. 

I purposefully set the hyperparameters to something absolutely stupid.  I also didnt do any clever feature engineering.  

TO USE: run fit_model() on the training data to generate trained_model.xgb, then zip the whole folder and upload as a submission
