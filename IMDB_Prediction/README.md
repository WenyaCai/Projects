# Predicting IMDb Scores for Unreleased Movies

This project focuses on developing a predictive model to estimate the IMDb scores for unreleased movies. The approach includes extensive data pre-processing, feature selection, model building, and tuning, using various regression techniques.

## Table of Contents
1. [Libraries and Dataset](#libraries-and-dataset)
    - [Libraries](#libraries)
    - [Dataset](#dataset)
2. [Data Pre-processing](#data-pre-processing)
    - [Drop Unnecessary Labels](#drop-unnecessary-labels)
    - [Categorical Data Processing](#categorical-data-processing)
        - [Factorizing Categorical Data](#factorizing-categorical-data)
        - [Handling Skewness in Categorical Data](#handling-skewness-in-categorical-data)
    - [Numerical Data Processing](#numerical-data-processing)
        - [Handling Skewness in Numerical Data](#handling-skewness-in-numerical-data)
3. [Feature Selection and Model Building](#feature-selection-and-model-building)
    - [Linear Regression](#linear-regression)
        - [Stepwise Variable Selection for Linear Regression](#stepwise-variable-selection-for-linear-regression)
        - [Linearity Test](#linearity-test)
    - [Non-linear Regression](#non-linear-regression)
        - [Model Fine Tuning](#model-fine-tuning)
        - [Handling Collinearity](#handling-collinearity)
            - [Collinearity Matrix](#collinearity-matrix)
            - [Variance Inflation Factors](#variance-inflation-factors)
            - [Re-Adjust Polynomial Degrees](#re-adjust-polynomial-degrees)
4. [Prediction on Test Data](#prediction-on-test-data)
    - [Pre-process Test Data](#pre-process-test-data)
    - [Fit the Model](#fit-the-model)
  
For more details, please refer to the `IMDB_Prediction.html` file.
