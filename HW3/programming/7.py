import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, make_scorer
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

parkinsons_data = pd.read_csv('C:\Apre\ML-Homeworks-1\HW3\programming\parkinsons.csv')

X = parkinsons_data.drop(columns=['target'])
y = parkinsons_data['target']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) #80% treino 20% teste

param_grid = {
    'alpha': [0.0001, 0.001, 0.01], # penalização L2
    'learning_rate_init': [0.001, 0.01, 0.1], # learning rate
    'batch_size': [32, 64, 128] # tamanho batch
}

mlp = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=0)

scorer = make_scorer(mean_absolute_error, greater_is_better=False)

grid_search = GridSearchCV(mlp, param_grid, cv=3, scoring=scorer, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_score = -grid_search.best_score_  # mete o MAE a positivo

best_model = grid_search.best_estimator_
y_pred_test = best_model.predict(X_test)
test_mae = mean_absolute_error(y_test, y_pred_test)

results = grid_search.cv_results_

# 3D plot
alphas = results['param_alpha'].data
learning_rates = results['param_learning_rate_init'].data
batch_sizes = results['param_batch_size'].data
mae_scores = -results['mean_test_score']

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')


sc = ax.scatter(alphas, learning_rates, batch_sizes, c=mae_scores, cmap='viridis') # Cria o scatter plot 3D

cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('MAE')

ax.set_xlabel('L2 Penalty (alpha)')
ax.set_ylabel('Learning Rate')
ax.set_zlabel('Batch Size')
ax.set_title('3D Plot of Hyperparameter Tuning (MAE)')

plt.show()

print("Best Hyperparameters:", best_params)
print("Best Cross-Validation MAE:", best_score)
print("Test MAE:", test_mae)
