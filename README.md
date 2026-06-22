# House Price Prediction

This project implements a linear regression model to predict house prices based on square footage, number of bedrooms, and number of bathrooms.

## Task
Part of the SkillCraft Technology Machine Learning Internship — Task 1.

## Dataset
[House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) (Kaggle)

## Features Used
- GrLivArea (square footage)
- BedroomAbvGr (number of bedrooms)
- FullBath + HalfBath (number of bathrooms)

## Approach
1. Loaded and cleaned the dataset using pandas
2. Split data into training and test sets
3. Trained a Linear Regression model using scikit-learn
4. Evaluated performance using R² score and RMSE
5. Visualized actual vs predicted prices with a scatter plot

## Results
- R² Score: 0.63
- RMSE: ~53,371

## Tools Used
- Python
- pandas
- scikit-learn
- matplotlib
