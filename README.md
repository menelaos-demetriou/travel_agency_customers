# Travel agency customers prediction
This repo contains the implementation of an estimator for customer prediction.
So far Logistic Regression and Random Forests are used. 
Accuracy is 82% and F1 score is 78%

Also a simple flask API is developed that accepts a row with customer attributes in json format
and returns a prediction if the customer will convert.

# To-do:
Add more models for grid search.
Try bayesian optimisation for hyperparameter tuning instead of grid search.
Add training and validation plots to examine bias-variance trade off.


# Requirements
pandas 
numpy  
scikit-learn  
seaborn  
matplotlib  
flask  
