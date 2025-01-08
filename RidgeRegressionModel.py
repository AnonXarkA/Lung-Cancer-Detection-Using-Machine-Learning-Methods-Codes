from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


X = cancer_patient.drop(["Level"], axis = 1)
y = cancer_patient["Level"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state= 40)


# Create Ridge Regression model
alpha = 0.5  # Regularization strength
ridge = Ridge(alpha=alpha)

# Train the Ridge Regression model
ridge.fit(X_train, y_train)

# Predict on the test set
y_pred = ridge.predict(X_test)

# Calculate R^2 score as accuracy metric
accuracy = r2_score(y_test, y_pred)
print(accuracy)


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# Calculate other performance metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

# Print the performance metrics
print("R^2 Score (Accuracy):", accuracy)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)

# Visualize predicted vs. actual values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Ridge Regression: Actual vs. Predicted Values")
plt.show()


