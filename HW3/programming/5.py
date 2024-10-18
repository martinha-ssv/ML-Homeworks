import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error


data = pd.read_csv('C:\Apre\ML-Homeworks-1\HW3\programming\parkinsons.csv')

X = data.drop(columns=['target'])  
y = data['target']  

mae_scores_lr = [] 
mae_scores_mlp_no_act = []
mae_scores_mlp_relu = []

for i in range(1, 11):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
    
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    mae_scores_lr.append(mae_lr)
    
    mlp_no_act = MLPRegressor(hidden_layer_sizes=(10, 10), activation='identity', max_iter=1000, random_state=0)
    mlp_no_act.fit(X_train, y_train)
    y_pred_mlp_no_act = mlp_no_act.predict(X_test)
    mae_mlp_no_act = mean_absolute_error(y_test, y_pred_mlp_no_act)
    mae_scores_mlp_no_act.append(mae_mlp_no_act)
    
    mlp_relu = MLPRegressor(hidden_layer_sizes=(10, 10), activation='relu', max_iter=1000, random_state=0)
    mlp_relu.fit(X_train, y_train)
    y_pred_mlp_relu = mlp_relu.predict(X_test)
    mae_mlp_relu = mean_absolute_error(y_test, y_pred_mlp_relu)
    mae_scores_mlp_relu.append(mae_mlp_relu)

mae_data = {
    'Linear Regression': mae_scores_lr,
    'MLP (No Activation)': mae_scores_mlp_no_act,
    'MLP (ReLU Activation)': mae_scores_mlp_relu
}

plt.figure(figsize=(8, 6))
sns.boxplot(data=[mae_scores_lr, mae_scores_mlp_no_act, mae_scores_mlp_relu])
plt.xticks([0, 1, 2], ['Linear Regression', 'MLP (No Activation)', 'MLP (ReLU Activation)'])
plt.ylabel('Test MAE')
plt.title('Boxplot of Test MAE for Linear Regression and MLP Regressors')
plt.grid(True)
plt.show()
