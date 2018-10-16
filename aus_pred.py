"""
This is the .py file for analysing house price in Austin
Author @ Chia-Hui Liu
"""
# import packages
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import seaborn as sns
import math
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.feature_selection import f_regression
import xgboost

def read_data(input_file):
    '''
    This is used for read the data
    '''
    df = pd.read_csv(input_file, encoding = 'iso-8859-1')
    # print the head of the data
    print(df.head())
    return df

def explortary_data(df):
    '''
    Data explortary
    '''
    # check the basic stats of data
    print(df.describe())
    # print info of each column
    print(df.info())
    # data preprocessing (drop na if existing)
    df = df.dropna(how='any')
    return df

def plot_scatter_matrix(df):
    '''
    Produce a scatterplot matrix which includes all of the variables in the data set.
    '''
    # initialize scatter_matrix with parameters
    df_scatter = scatter_matrix(df, alpha=0.03, figsize=(13,13), diagonal='hist')
    # Change label rotation
    for x_label in df_scatter.reshape(-1):
        x_label.xaxis.label.set_rotation(45)
    for y_label in df_scatter.reshape(-1):
        y_label.yaxis.label.set_rotation(-45)
    # solve visualization overlapping
    for y_label in df_scatter.reshape(-1):
        y_label.get_yaxis().set_label_coords(-1.2, 0.5)
    # show the plot
    plt.show()

def plot_corr_matrix(df):
    '''
    Compute the matrix of the correlations between the variables using the function
    '''
    # calculate correlations
    correlations = df.corr()
    # print correlation results
    print(correlations)
    # plot correlations by matrix
    fig, ax = plst.subplots(figsize=(15, 15))
    # plot the correlation heatmap
    sns.heatmap(correlations)

def linear_model_original(true_y_salePrice, x_variable):
    '''
    Generate linear regression model with all features with original format.
    '''
    print('Generate linear regression model with all features with original format.')
    # separate the test data and train data (train : test = 80%:20%)
    X_train, X_test, y_train, y_test = train_test_split(x_variable, true_y_salePrice, test_size=0.2, random_state = 1)
    # results
    print('Coefficients: \n', str(model_res.coef_))
    print('Intercept: ' + str(model_res.intercept_))
    print('MSE: ' + str(mean_squared_error(y_test, predictions)))
    # MSE can also computed as np.mean((predictions - y_test) ** 2
    print('R Square: ' + str(r2_score(y_test, predictions)))

def transform_sqrt(true_y_salePrice, x_variable):
    """
    transformed the data by sqared root
    """
    print('transforming data by sqrt')
    transformeed_squrtx = x_variable.transform(lambda x: np.sqrt(x))
    # separate the test data and train data (train : test = 80%:20%)
    X_train, X_test, y_train, y_test = train_test_split(transformeed_squrtx, true_y_salePrice, test_size=0.2, random_state = 1)
    sqrt_ols = linear_model.LinearRegression()
    sqrt_res = sqrt_ols.fit(X_train, y_train)
    y_pred = sqrt_ols.predict(X_test)
    # print the results
    print('Coefficients: \n', str(sqrt_res.coef_))
    print('Intercept: ' + str(sqrt_res.intercept_))
    print('MSE: ' + str(mean_squared_error(y_test, y_pred)))
    # MSE can also computed as np.mean((y_pred - y_test) ** 2
    print('R Square: ' + str(r2_score(y_test, y_pred)))

def transform_square(true_y_salePrice, x_variable):
    # transformed the data by n**2
    print()
    print('Transforming data by square')
    transformeed_squarex = x_variable.transform(lambda x: x**2)
    # separate the test data and train data (train : test = 80%:20%)
    X_train, X_test, y_train, y_test = train_test_split(transformeed_squarex, true_y_salePrice, test_size=0.2, random_state = 1)
    sq_ols = linear_model.LinearRegression()
    sq_res = sq_ols.fit(X_train, y_train)
    y_pred = sq_ols.predict(X_test)
    # print results
    print('Coefficients: \n', str(sq_res.coef_))
    print('Intercept: ' + str(sq_res.intercept_))
    print('MSE: ' + str(mean_squared_error(y_test, y_pred)))
    # MSE can also computed as np.mean((y_pred - y_test) ** 2
    print('R Square: ' + str(r2_score(y_test, y_pred)))

def xgb_regressor(true_y_salePrice, x_variable):
    '''
    Using XGBoost regressor
    '''
    xgb = xgboost.XGBRegressor(n_estimators=500, learning_rate=0.1, gamma=0, subsample=0.75, colsample_bytree=1, max_depth=8)
    # separate the test data and train data (train : test = 80%:20%)
    X_train, X_test, y_train, y_test = train_test_split(x_variable, true_y_salePrice, test_size=0.2, random_state = 1)
    # fit data
    xgb.fit(X_train,y_train)
    y_pred = xgb.predict(X_test)

    print('MSE: ' + str(mean_squared_error(y_test, y_pred)))
    # MSE can also computed as np.mean((y_pred - y_test) ** 2
    print('R Square: ' + str(r2_score(y_test, y_pred)))

if __name__ == "__main__":
    aus_house_price_df = read_data('austin_house_price.csv')
    aus_house_price_df = explortary(aus_house_price_df)
    plot_scatter_matrix(aus_house_price_df)
    plot_corr_matrix(aus_house_price_df)
    # the true y
    true_y_salePrice = df['SalePrice']
    # keep all variables as x except for the SalePrice
    x_variable = df.drop('SalePrice',axis=1)
    # linear regression model
    linear_model_original(true_y_salePrice, x_variable)
    # transform data by squared root
    transform_sqrt(true_y_salePrice, x_variable)
    # transform data by sqaring
    transform_square(true_y_salePrice, x_variable)
    # using xgboost regressor
    xgb_regressor(true_y_salePrice, x_variable)
