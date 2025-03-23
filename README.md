# Chinese GP 2025 Race Results Prediction

This project predicts the race results for the 2025 Chinese Grand Prix (held on March 23, 2025) using historical Formula 1 data and machine learning. We use a Random Forest Regressor to predict the finishing positions of drivers based on their qualifying positions, team performance, and other features.

## Project Overview

- **Goal**: Predict the finishing positions of drivers in the Chinese GP 2025 race using historical Formula 1 data.
- **Dataset**: Historical qualifying and race results from 2024 and 2025 (simulated for this project), combined with manually created qualifying data for the 2025 Chinese GP.
- **Model**: Random Forest Regressor from scikit-learn.
- **Features Used**:
  - Qualifying Position
  - Starting Grid Position
  - Driver and Team (encoded)
  - Track (encoded)
  - Average finish position of drivers and teams (feature engineering)
- **Evaluation Metrics**:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)

## Prerequisites

To run this project, you need the following Python libraries:
- `pandas`
- `scikit-learn`
- `numpy`
- `matplotlib`
- `seaborn`


Steps and Code Explanation
--------------------------

### 1\. Import Libraries

We import the necessary libraries for data manipulation (pandas, numpy), machine learning (scikit-learn), and visualization (matplotlib, seaborn).

pythonCollapseWrapCopyimport pandas as pdimport numpy as npfrom sklearn.model\_selection import train\_test\_splitfrom sklearn.metrics import mean\_squared\_error, mean\_absolute\_errorfrom sklearn.preprocessing import LabelEncoderfrom sklearn.ensemble import RandomForestRegressorimport matplotlib.pyplot as pltimport seaborn as sns

### 2\. Create 2025 Chinese GP Qualifying Data

We manually create a dataset for the 2025 Chinese GP qualifying results, including driver names, teams, and their Q1, Q2, and Q3 times. This data is stored in a pandas DataFrame.

### 3\. Load Historical Data

We define a function load\_data to load historical qualifying and race results from 2024 and 2025 (simulated). The data is fetched from a GitHub repository and concatenated into a single DataFrame for training.

### 4\. Data Preprocessing

*   **Merging Data**: We merge race and qualifying data based on Year, Track, Driver, and Team.
    
*   **Handling Missing Values**: Non-numeric race results (e.g., "DNF", "Retired") are replaced with NaN, and rows with missing finish positions are dropped.
    
*   **Encoding Categorical Variables**: LabelEncoder is used to encode Driver, Team, and Track into numerical values.
    
*   **Feature Engineering**: We calculate the average finish position for each driver and team to use as additional features.
    

### 5\. Model Training

*   **Features**: We use Qualifying\_Position, Starting Grid, Driver\_Encoded, Team\_Encoded, Track\_Encoded, Driver\_Avg\_Finish, and Team\_Avg\_Finish as features.
    
*   **Target**: The target variable is Finish\_Position.
    
*   **Model**: A RandomForestRegressor with 100 estimators is trained on the data.
    
*   **Evaluation**: The model is evaluated using MAE and MSE on a test set (20% of the data).
    

### 6\. Predict Chinese GP 2025 Results

*   We prepare the 2025 Chinese GP data by encoding categorical variables and adding engineered features.
    
*   The trained Random Forest model predicts the finishing positions for each driver.
    
*   The results are sorted by predicted finish position to create a predicted ranking.
    

### 7\. Visualize Feature Importance

We use a bar plot to visualize the importance of each feature in the Random Forest model.
