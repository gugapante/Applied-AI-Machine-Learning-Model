import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Define the best parameter grid
best_mlp_params = {
    'activation': 'relu',
    'batch_size': 128,
    'hidden_layer_sizes': (150, 100),
    'learning_rate': 'adaptive',
    'max_iter': 1000,
    'solver': 'lbfgs'
}

# Step 1: Load the data set:
data = pd.read_csv('boston.csv')
# Remove any loading/trailing whitespace
data.columns = data.columns.str.strip()
# Print the first 5 rows of the data
print(data.head())

# Step 2: Define the features and target variable:
X = data.drop(columns=['MedV'])
Y = data['MedV']

# Set up the cross-validation using ShuffleSplit
cv = ShuffleSplit(n_splits=5, test_size=0.2)

# Step 3: Initialise the storage for the performance data
lr_mae_data, rf_mae_data, gb_mae_data, mlp_mae_data = [], [], [], []
lr_mse_data, rf_mse_data, gb_mse_data, mlp_mse_data = [], [], [], []
lr_rmse_data, rf_rmse_data, gb_rmse_data, mlp_rmse_data = [], [], [], []
lr_r2_data, rf_r2_data, gb_r2_data, mlp_r2_data = [], [], [], []

# Step 4: Set up a for loop to run the model a total of 5 times:
for i in range(5):
    # Split the data:
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    # Step 5: Initialise and train the linear regression model
    lr_model = LinearRegression()
    lr_model.fit(X_train, Y_train)

    # Make predictions on the test data for Linear Regression
    Y_lr_predict = lr_model.predict(X_test)

    # Evaluate the linear regression model
    lr_mae = mean_absolute_error(Y_test, Y_lr_predict)
    lr_mse = mean_squared_error(Y_test, Y_lr_predict)
    lr_rmse = np.sqrt(lr_mse)
    lr_r2 = r2_score(Y_test, Y_lr_predict)

    # Initialise and train the Random Forest model
    rf_model = RandomForestRegressor()
    rf_model.fit(X_train, Y_train)

    # Make predictions on the test data for Random Forest Regressor
    Y_rf_predict = rf_model.predict(X_test)

    # Evaluate the Random Forest model
    rf_mae = mean_absolute_error(Y_test, Y_rf_predict)
    rf_mse = mean_squared_error(Y_test, Y_rf_predict)
    rf_rmse = np.sqrt(rf_mse)
    rf_r2 = r2_score(Y_test, Y_rf_predict)

    # Initialise and train the Gradient Boosting model
    gb_model = GradientBoostingRegressor()
    gb_model.fit(X_train, Y_train)

    # Make predictions on test data for Gradient Boosting Regressor
    Y_gb_predict = gb_model.predict(X_test)

    # Evaluate the Gradient Boosting model
    gb_mae = mean_absolute_error(Y_test, Y_gb_predict)
    gb_mse = mean_squared_error(Y_test, Y_gb_predict)
    gb_rmse = np.sqrt(gb_mse)
    gb_r2 = r2_score(Y_test, Y_gb_predict)

    # Initialise and train the MLP model
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    mlp_model = MLPRegressor(**best_mlp_params)
    mlp_model.fit(X_train_scaled, Y_train)

    # Make predictions on test data for the MLP Regressor
    Y_mlp_predict = mlp_model.predict(X_test_scaled)

    # Evaluate the MLP model
    mlp_mae = mean_absolute_error(Y_test, Y_mlp_predict)
    mlp_mse = mean_squared_error(Y_test, Y_mlp_predict)
    mlp_rmse = np.sqrt(mlp_mse)
    mlp_r2 = r2_score(Y_test, Y_mlp_predict)

    # Save the results for each of the runs and each of the models
    lr_mae_data.append(lr_mae)
    lr_mse_data.append(lr_mse)
    lr_rmse_data.append(lr_rmse)
    lr_r2_data.append(lr_r2)

    rf_mae_data.append(rf_mae)
    rf_mse_data.append(rf_mse)
    rf_rmse_data.append(rf_rmse)
    rf_r2_data.append(rf_r2)

    gb_mae_data.append(gb_mae)
    gb_mse_data.append(gb_mse)
    gb_rmse_data.append(gb_rmse)
    gb_r2_data.append(gb_r2)

    mlp_mae_data.append(mlp_mae)
    mlp_mse_data.append(mlp_mse)
    mlp_rmse_data.append(mlp_rmse)
    mlp_r2_data.append(mlp_r2)

    # Perform cross-validation for Mean Squared Error (negated) and R^2
    lr_mae_scores = cross_val_score(lr_model, X, Y, cv=cv, scoring='neg_mean_absolute_error')
    lr_mse_scores = cross_val_score(lr_model, X, Y, cv=cv, scoring='neg_mean_squared_error')
    lr_r2_scores = cross_val_score(lr_model, X, Y, cv=cv, scoring='r2')

    # Convert MSE and MAE scores to positive values and calculate RMSE from MSE
    lr_mae_scores = -lr_mae_scores
    lr_mse_scores = -lr_mse_scores  # Negate to make MSE positive
    lr_rmse_scores = np.sqrt(lr_mse_scores)  # RMSE is the square root of MSE

    rf_mae_scores = cross_val_score(rf_model, X, Y, cv=cv, scoring='neg_mean_absolute_error')
    rf_mse_scores = cross_val_score(rf_model, X, Y, cv=cv, scoring='neg_mean_squared_error')
    rf_r2_scores = cross_val_score(rf_model, X, Y, cv=cv, scoring='r2')
    rf_mae_scores = -rf_mae_scores
    rf_mse_scores = -rf_mse_scores
    rf_rmse_scores = np.sqrt(rf_mse_scores)

    gb_mae_scores = cross_val_score(gb_model, X, Y, cv=cv, scoring='neg_mean_absolute_error')
    gb_mse_scores = cross_val_score(gb_model, X, Y, cv=cv, scoring='neg_mean_squared_error')
    gb_r2_scores = cross_val_score(gb_model, X, Y, cv=cv, scoring='r2')
    gb_mae_scores = -gb_mae_scores
    gb_mse_scores = -gb_mse_scores
    gb_rmse_scores = np.sqrt(gb_mse_scores)

    mlp_mae_scores = cross_val_score(mlp_model, X, Y, cv=cv, scoring='neg_mean_absolute_error')
    mlp_mse_scores = cross_val_score(mlp_model, X, Y, cv=cv, scoring='neg_mean_squared_error')
    mlp_r2_scores = cross_val_score(mlp_model, X, Y, cv=cv, scoring='r2')
    mlp_mae_scores = -mlp_mae_scores
    mlp_mse_scores = -mlp_mse_scores
    mlp_rmse_scores = np.sqrt(mlp_mse_scores)


# Step 6: Print the results out in the console window to better analyse the results
print('\nLinear Regression:')
print(f'Mean Absolute Error: {lr_mae}')
print(f'Mean Squared Error: {lr_mse}')
print(f'Root Mean Squared Error: {lr_rmse}')
print(f'R\u00B2 Score: {lr_r2}')

print('\nRandom Forest Regression:')
print(f'Mean Absolute Error: {rf_mae}')
print(f'Mean Squared Error: {rf_mse}')
print(f'Root Mean Squared Error: {rf_rmse}')
print(f'R\u00B2 Score: {rf_r2}')

print('\nGradient Boosting Regression:')
print(f'Mean Absolute Error: {gb_mae}')
print(f'Mean Squared Error: {gb_mse}')
print(f'Root Mean Squared Error: {gb_rmse}')
print(f'R\u00B2 Score: {gb_r2}')

print('\nMLP Regressor Model:')
print(f'Mean Absolute Error: {mlp_mae}')
print(f'Mean Squared Error: {mlp_mse}')
print(f'Root Mean Squared Error: {mlp_rmse}')
print(f'R\u00B2 Score: {mlp_r2}')

# Print the results for cross-validation of all three models to see the performance
# on each fold of the cross-validation process
print('\nLinear Regression Cross-Validation Mean Absolute Error Scores: ', lr_mae_scores)
print('Linear Regression Cross-Validation Mean Squared Error Scores: ',lr_mse_scores)
print('Linear Regression Cross-Validation Root Mean Squared Error Scores: ',lr_rmse_scores)
print('Linear Regression Cross-Validation R\u00B2 Scores: ',lr_r2_scores)

print('\nRandom Forest Cross-Validation Mean Absolute Error Scores: ', rf_mae_scores)
print('Random Forest Cross-Validation Mean Squared Error Scores: ', rf_mse_scores)
print('Random Forest Cross-Validation Root Mean Squared Error Scores: ', rf_rmse_scores)
print('Random Forest Cross-Validation R\u00B2 Scores: ', rf_r2_scores)

print('\nGradient Boosting Cross-Validation Mean Absolute Error Scores: ', gb_mae_scores)
print('Gradient Boosting Cross-Validation Mean Squared Error Scores: ', gb_mse_scores)
print('Gradient Boosting Cross-Validation Root Mean Squared Error Scores: ', gb_rmse_scores)
print('Gradient Boosting Cross-Validation R\u00B2 Scores: ', gb_r2_scores)

print('\nMLP Regressor Cross-Validation Mean Absolute Error Scores: ', mlp_mae_scores)
print('MLP Regressor Cross-Validation Mean Squared Error Scores: ', mlp_mse_scores)
print('MLP Regressor Cross-Validation Root Mean Squared Error Scores: ', mlp_rmse_scores)
print('MLP Regressor Cross-Validation R\u00B2 Scores: ', mlp_r2_scores)

# Print summary of overall performance across all folds by calculating
# the average of the individual scores
print('\nAverage Linear Regression Cross-Validation MAE: ',lr_mae_scores.mean())
print('Average Linear Regression Cross-Validation MSE: ',lr_mse_scores.mean())
print('Average Linear Regression Cross-Validation RMSE: ',lr_rmse_scores.mean())
print('Average Linear Regression Cross-Validation R\u00B2: ',lr_r2_scores.mean())

print('\nAverage Random Forest Cross-Validation MAE: ',rf_mae_scores.mean())
print('Average Random Forest Cross-Validation MSE: ', rf_mse_scores.mean())
print('Average Random Forest Cross-Validation RMSE: ', rf_rmse_scores.mean())
print('Average Random Forest Cross-Validation R\u00B2: ', rf_r2_scores.mean())

print('\nAverage Gradient Boosting Cross-Validation MAE: ',gb_mae_scores.mean())
print('Average Gradient Boosting Cross-Validation MSE: ', gb_mse_scores.mean())
print('Average Gradient Boosting Cross-Validation RMSE: ', gb_rmse_scores.mean())
print('Average Gradient Boosting Cross-Validation R\u00B2: ', gb_r2_scores.mean())

print('\nAverage MLP Regressor Cross-Validation MAE: ',mlp_mae_scores.mean())
print('Average MLP Regressor Cross-Validation MSE: ', mlp_mse_scores.mean())
print('Average MLP Regressor Cross-Validation RMSE: ', mlp_rmse_scores.mean())
print('Average MLP Regressor Cross-Validation R\u00B2: ', mlp_r2_scores.mean())

# Step 7: Plot the results for each of the runs for each data set
plt.figure(figsize=(12, 6))
plt.plot(range(1, 6), lr_mae_data, label='Linear Regression MAE', marker='o')
plt.plot(range(1, 6), rf_mae_data, label='Random Forest Regressor MAE', marker='o')
plt.plot(range(1, 6), gb_mae_data, label='Gradient Boosting Regressor MAE', marker='o')
plt.plot(range(1, 6), mlp_mae_data, label='MLP Regressor MAE', marker='o')
plt.title('MAE Comparison Across 5 Runs')
plt.xlabel('Run')
plt.ylabel('MAE')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(range(1, 6), lr_mse_data, label='Linear Regression MSE', marker='o')
plt.plot(range(1, 6), rf_mse_data, label='Random Forest Regressor MSE', marker='o')
plt.plot(range(1, 6), gb_mse_data, label='Gradient Boosting Regressor MSE', marker='o')
plt.plot(range(1, 6), mlp_mse_data, label='MLP Regressor MSE', marker='o')
plt.title('MSE Comparison Across 5 Runs')
plt.xlabel('Run')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.show()

# Plot RMSE for each model
plt.figure(figsize=(12, 6))
plt.plot(range(1, 6), lr_rmse_data, label='Linear Regression RMSE', marker='o')
plt.plot(range(1, 6), rf_rmse_data, label='Random Forest Regressor RMSE', marker='o')
plt.plot(range(1, 6), gb_rmse_data, label='Gradient Boosting Regressor RMSE', marker='o')
plt.plot(range(1, 6), mlp_rmse_data, label='MLP Regressor RMSE', marker='o')
plt.title('RMSE Comparison Across 5 Runs')
plt.xlabel('Run')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True)
plt.show()

# Plot R^2 for each model
plt.figure(figsize=(12, 6))
plt.plot(range(1, 6), lr_r2_data, label='Linear Regression R\u00B2', marker='o')
plt.plot(range(1, 6), rf_r2_data, label='Random Forest R\u00B2', marker='o')
plt.plot(range(1, 6), gb_r2_data, label='Gradient Boosting R\u00B2', marker='o')
plt.plot(range(1, 6), mlp_r2_data, label='MLP Regressor R\u00B2', marker='o')
plt.title('R\u00B2 Comparison Across 5 Runs')
plt.xlabel('Run')
plt.ylabel('R\u00B2')
plt.legend()
plt.grid(True)
plt.show()