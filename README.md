# Fuel Efficiency Prediction with Neural Networks

This repository contains code for predicting fuel efficiency (FE) using a neural network built with TensorFlow/Keras. The code utilizes data preprocessing techniques from scikit-learn and demonstrates a basic neural network architecture for regression.

## Overview

The script `fuel_efficiency_prediction.py` performs the following steps:

1.  **Data Loading:** Reads the `fuel.csv` dataset using pandas.
2.  **Data Preprocessing:**
    * Separates the target variable ('FE') from the features.
    * Applies `StandardScaler` to numerical features.
    * Applies `OneHotEncoder` to categorical features.
    * Performs a log transformation on the target variable.
3.  **Neural Network Model:**
    * Builds a sequential neural network with dense layers using ReLU activation.
    * Compiles the model with the Adam optimizer and mean absolute error (MAE) loss.
4.  **Model Training:** Trains the model on the preprocessed data.
5.  **SGD Animation:** Includes a call to a function `animate_sgd` which is assumed to be defined elsewhere. This function likely visualizes the Stochastic Gradient Descent process. (Note: The `animate_sgd` function is not fully defined in the provided code snippet, so this README assumes its existence and purpose based on the function call.)

## Dependencies

* Python 3.x
* pandas
* numpy
* scikit-learn
* tensorflow

You can install the required dependencies using pip:

```bash
pip install pandas numpy scikit-learn tensorflow
