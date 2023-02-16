# scatter plot of RCFM and FM colored by site
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
import matplotlib.font_manager as font_manager

# read the xlsx file
df = pd.read_excel('/Users/renyuxuan/Desktop/Research/RCFM_FT-IR_forPython.xlsx', sheet_name='RCFM_imposing')
X = df['FM']
y1 = df['RCFM']
y2 = df['RCFM_imposing']

# reshape to have one column
X = X.values.reshape(-1, 1)
y1 = y1.values.reshape(-1, 1)
y2 = y2.values.reshape(-1, 1)

# impute the missing values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)
y1 = imputer.fit_transform(y1)
y2 = imputer.fit_transform(y2)

# Check for NaN values
print(np.isnan(X).any())  # Should return False if there are no NaN values
print(np.isnan(y1).any())  # Should return False if there are no NaN values
print(np.isnan(y2).any())  # Should return False if there are no NaN values

# calculate the regression equation and r-square for RCFM
reg_1 = LinearRegression().fit(X, y1)
slope_1 = reg_1.coef_[0]
intercept_1 = reg_1.intercept_
slope_1_scalar = slope_1[0]
intercept_1_scalar = intercept_1[0]
print(f"Regression equation for RCFM: y = {slope_1_scalar:.2f}x + {intercept_1_scalar:.2f}")
y_pred_1 = reg_1.predict(X)
r2_1 = r2_score(y1, y_pred_1)
print(f"R-squared for RCFM: {r2_1:.2f}")

reg_2 = LinearRegression().fit(X, y2)
slope_2 = reg_2.coef_[0]
intercept_2 = reg_2.intercept_
slope_2_scalar = slope_2[0]
intercept_2_scalar = intercept_2[0]
print(f"Regression equation for RCFM_imposing: y = {slope_2_scalar:.2f}x + {intercept_2_scalar:.2f}")
y_pred_2 = reg_2.predict(X)
r2_2 = r2_score(y2, y_pred_2)
print(f"R-squared for RCFM_imposing: {r2_2:.2f}")

