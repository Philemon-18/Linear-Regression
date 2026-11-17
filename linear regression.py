import numpy as np 
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from scipy.stats import pearsonr

# This data ranges from 1/1/2010 to 1/10/2023

Inflation_rates = np.array([0.64, 0.39, 0.61, 0.90, 0.79, 1.20, 0.91, 0.58, 0.80, 0.71, 1.83, 0.61, 2.13, 2.37, 1.36, 1.27, 1.33, 1.66, 2.41, 3.39, 1.64, 2.08, 1.82, 3.13, 2.95, 1.20, 2.64, 2.09, 1.29, 2.26, 2.67, 2.31, 2.21, 1.92, 1.69, 2.06, 2.45, 2.13, 2.07, 0.13, 0.97, 1.89, 2.56, 2.87, 1.93, 1.67, 1.98, 1.84, 1.85, 1.74, 2.46, 2.57, 2.18, 2.07, 2.03, 1.20, 1.82, 2.28, 1.96, 1.38, 2.03, 1.69, 2.67, 3.35, 2.22, 2.64, 2.35, 1.66, 3.02, 3.09, 2.66, 1.89, 2.31, 2.62, 2.52, 3.41, 2.63, 2.62, 2.42, 3.42, 1.92, 1.62, 2.86, 2.69, 3.11, 2.77, -0.17, 1.00, 1.48, 2.41, 2.54, 2.73, 2.12, 2.66, 1.92, 2.90, 3.60, 2.35, 2.72, 2.49, 2.09, 2.51, 2.14, 1.70, 2.48, 2.08, 2.66, 2.66, 2.62, 2.36, 2.50, 2.92, 1.71, 2.89, 3.15, 3.55, 2.39, 3.58, 2.95, 2.42, 3.19, 2.27, 0.13, -2.16, -0.19, 2.87, 6.70, 2.76, -0.34, 0.67, 2.44, 1.18, 0.59, 2.63, 2.37, 5.23, 3.92, 2.81, 3.73, 2.72, 2.96, 5.67, 4.56, 4.32, 6.67, 6.50, 5.47, 7.26, 6.99, 8.06, 5.98, 6.91, 7.65, 5.43, 5.98, 5.68, 6.00, 6.97, 3.89, 5.72, 3.58, 3.27, 3.78, 4.09, 5.09, 4.63, 3.74, 4.08, 6.50, 4.56, 5.16, 4.51, 2.37, 2.44, 3.08, 3.25, 3.88, 4.19, 2.74, 2.44, 4.93, 3.31, 2.09, 3.45, 2.24, 4.06, 4.58, 3.29, 2.58, 1.27, 1.27, 3.06, 4.61, 2.41, 2.39, 4.78, 2.87, 0.67, 3.48, 3.11, 3.49, 4.67])
Interest_rates = np.array([0.11, 0.13, 0.16, 0.20, 0.20, 0.18, 0.18, 0.19, 0.19, 0.19, 0.19, 0.18, 0.17, 0.16, 0.14, 0.10, 0.09, 0.09, 0.07, 0.10, 0.08, 0.07, 0.08, 0.07, 0.08, 0.10, 0.13, 0.14, 0.16, 0.16, 0.16, 0.13, 0.14, 0.16, 0.16, 0.16, 0.14, 0.15, 0.14, 0.15, 0.11, 0.09, 0.09, 0.08, 0.08, 0.09, 0.08, 0.09, 0.07, 0.07, 0.08, 0.09, 0.09, 0.10, 0.09, 0.09, 0.09, 0.09, 0.09, 0.12, 0.11, 0.11, 0.11, 0.12, 0.12, 0.13, 0.13, 0.14, 0.14, 0.12, 0.12, 0.24, 0.34, 0.38, 0.36, 0.37, 0.37, 0.38, 0.39, 0.40, 0.40, 0.40, 0.41, 0.54, 0.65, 0.66, 0.79, 0.90, 0.91, 1.04, 1.15, 1.16, 1.15, 1.15, 1.16, 1.30, 1.41, 1.42, 1.51, 1.69, 1.70, 1.82, 1.91, 1.91, 1.95, 2.19, 2.20, 2.27, 2.40, 2.40, 2.41, 2.42, 2.39, 2.38, 2.40, 2.13, 2.04, 1.83, 1.55, 1.55, 1.55, 1.58, 0.65, 0.05, 0.05, 0.08, 0.09, 0.10, 0.09, 0.09, 0.09, 0.09, 0.09, 0.08, 0.07, 0.07, 0.06, 0.08, 0.10, 0.09, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.20, 0.33, 0.77, 1.21, 1.68, 2.33, 2.56, 3.08, 3.78, 4.10, 4.33, 4.57, 4.65, 4.83, 5.06, 5.08, 5.12, 5.33, 5.33, 5.33, 5.33, 5.33, 5.33, 5.33, 5.33, 5.33, 5.33, 5.33, 5.33, 5.33, 5.13, 4.83, 4.64, 4.48, 4.33, 4.33, 4.33, 4.33, 4.33, 4.33, 4.33, 4.33, 4.22, 4.09, 3.88, 3.72, 3.64, 3.64, 3.64, 3.64, 3.63, 3.63, 3.63, 3.63, 3.75, 3.88])

w, b, lr = 0.0, 0.0, 0.02 # Weight and bias both start from 0

for epoch in range(1000):
    y_hat = Inflation_rates * w + b          # Prdiction equation y_hat is the predicted value of Interest_rates
    error = y_hat - Interest_rates    # Residuals, how far are the predicted values are away from the actual values
    dw = 2*(error*Inflation_rates).mean()   # Gradient, tells which way is downhill
    db = 2*error.mean()
    w -= lr * dw              # takes a small step down
    b -= lr * db 


plt.scatter(Inflation_rates, Interest_rates, label="Observed data")
plt.scatter(Inflation_rates, y_hat, label="Predicted data", alpha=0.5)
plt.plot(Inflation_rates, y_hat, label="Fitted Line: y = 0.4659X +  0.2303",color="red")
plt.xlabel("Inflation Rates")
plt.ylabel("Interest Rates")
plt.title("Linear Regression Using Gradient Descent")
plt.legend()
plt.grid(True)
plt.show()

print(f"Weight:{w}, Bias: {b}")

#Calculate the correlation coefficient and coefficient of determination between Interest_rates and Inflation_rates
r, p_value = pearsonr(Interest_rates, Inflation_rates)
r_squared = r**2
print(f"P-value: {p_value}")
print(f"Correlation Coefficient: {r}")
print(f"Coefficient of Determination: {r_squared}")