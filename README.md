# Chinese GP 2025 Race Results Prediction

This project predicts the race results for the 2025 Chinese Grand Prix (held on March 23, 2025) using historical Formula 1  data from toUpperCase78/formula1-datasets repository and machine learning. We use a Random Forest Regressor to predict the finishing positions of drivers based on their qualifying positions, team performance, and other features.

Goal: Predict driver finishing positions for the Chinese GP 2025
Data: Historical 2024-2025 F1 results + 2025 Chinese GP qualifying data
Model: Random Forest Regressor
Key Features: Qualifying position, team performance, driver history

# Requirements
pip install pandas scikit-learn numpy matplotlib seaborn

# Data Processing
Load historical data
  historical_data = load_data()
# Prepare Chinese GP 2025 data
 chinese_gp_data = prepare_gp_data(qualifying_data)
# Preprocess data
 Replace 'DNF' with NaN
 Encode categorical variables
 Engineer features (driver/team averages)
 X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)
# Train Model
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)
# Evaluate model
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
Predict Race Results
# Generate and sort predictions
predicted_positions = model.predict(chinese_gp_data)
results = pd.DataFrame({'Driver': drivers, 'Predicted_Position': predicted_positions})
results.sort_values('Predicted_Position').reset_index(drop=True)
# Visualize Results
Plot feature importance

feature_importance = pd.DataFrame({'Feature': feature_names,'Importance': model.feature_importances_})

sns.barplot(x='Importance', y='Feature', data=feature_importance.sort_values('Importance'))
# Evaluation
The model performance is measured using Mean Absolute Error (MAE) and Mean Squared Error (MSE) metrics, providing insight into prediction accuracy.
