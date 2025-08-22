# Chinese GP 2025 Race Results Prediction Model

This project predicts Formula 1 race finishing positions for the 2025 Chinese Grand Prix using a Random Forest Regressor model. It leverages historical race and qualifying data, combined with feature engineering, to forecast driver performance based on qualifying results, driver and team averages, and other relevant factors.

### Goal: Predict driver finishing positions for the Chinese GP 2025

### Data: Historical 2024-2025 F1 results + 2025 Chinese GP qualifying data

### Model: Random Forest Regressor

### Features

- Qualifying position and starting grid.
- Encoded driver, team, and track information.
- Average driver and team finishing positions from historical data.

### Evaluation Metrics:

- Mean Absolute Error (MAE): 2.45 positions
- Mean Squared Error (MSE): 10.24

### Analysis
- Podium Accuracy: The model perfectly predicted the top three positions, correctly identifying Piastri, Norris, and Russell in their exact order.
- Top 4 Accuracy: The model also correctly predicted Max Verstappen in 4th place, making the top four predictions accurate.
