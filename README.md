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

Install them using the following command:
```bash
pip install pandas scikit-learn numpy matplotlib seaborn
Steps and Code Explanation
1. Import Libraries
We import the necessary libraries for data manipulation (pandas, numpy), machine learning (scikit-learn), and visualization (matplotlib, seaborn).

python

Collapse

Wrap

Copy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
2. Create 2025 Chinese GP Qualifying Data
We manually create a dataset for the 2025 Chinese GP qualifying results, including driver names, teams, and their Q1, Q2, and Q3 times. This data is stored in a pandas DataFrame.

3. Load Historical Data
We define a function load_data to load historical qualifying and race results from 2024 and 2025 (simulated). The data is fetched from a GitHub repository and concatenated into a single DataFrame for training.

4. Data Preprocessing
Merging Data: We merge race and qualifying data based on Year, Track, Driver, and Team.
Handling Missing Values: Non-numeric race results (e.g., "DNF", "Retired") are replaced with NaN, and rows with missing finish positions are dropped.
Encoding Categorical Variables: LabelEncoder is used to encode Driver, Team, and Track into numerical values.
Feature Engineering: We calculate the average finish position for each driver and team to use as additional features.
5. Model Training
Features: We use Qualifying_Position, Starting Grid, Driver_Encoded, Team_Encoded, Track_Encoded, Driver_Avg_Finish, and Team_Avg_Finish as features.
Target: The target variable is Finish_Position.
Model: A RandomForestRegressor with 100 estimators is trained on the data.
Evaluation: The model is evaluated using MAE and MSE on a test set (20% of the data).
6. Predict Chinese GP 2025 Results
We prepare the 2025 Chinese GP data by encoding categorical variables and adding engineered features.
The trained Random Forest model predicts the finishing positions for each driver.
The results are sorted by predicted finish position to create a predicted ranking.
7. Visualize Feature Importance
We use a bar plot to visualize the importance of each feature in the Random Forest model.

Predicted vs Actual Results
Below is a comparison between the predicted rankings (from the qualifying data) and the actual race results (from the model predictions).

Driver	Predicted Rank	Result Rank
Oscar Piastri	1	1
Lando Norris	2	2
George Russell	3	3
Max Verstappen	4	4
Lewis Hamilton	5	6
Charles Leclerc	6	5
Isack Hadjar	7	14
Kimi Antonelli	8	8
Carlos Sainz	9	13
Alex Albon	10	9
Yuki Tsunoda	11	19
Fernando Alonso	12	DNF
Nico Hulkenberg	13	18
Oliver Bearman	14	10
Esteban Ocon	15	7
Pierre Gasly	16	11
Gabriel Bortoleto	17	17
Lance Stroll	18	12
Jack Doohan	19	16
Liam Lawson	20	15
Notes on Comparison
Accurate Predictions: Drivers like Oscar Piastri, Lando Norris, George Russell, Max Verstappen, and Kimi Antonelli had their positions predicted accurately or very closely.
Discrepancies: Some drivers, such as Isack Hadjar (predicted 7th, finished 14th) and Yuki Tsunoda (predicted 11th, finished 19th), had significant differences, possibly due to race incidents or model limitations.
DNF: Fernando Alonso was predicted to finish 12th but did not finish (DNF) the race.
How to Run the Code
Clone this repository.
Install the required libraries using the command above.
Run the Python script to generate predictions and visualize feature importance.
Check the console output for predicted rankings and the comparison table in this README for a summary.
Future Improvements
Incorporate more features, such as weather conditions, tire strategy, or pit stop data.
Use a more advanced model, like a neural network, for better prediction accuracy.
Collect more historical data to improve the model's training.
License
This project is licensed under the MIT License.

text

Collapse

Wrap

Copy

---

### Instructions for GitHub
1. Copy the entire Markdown code block above.
2. Go to your GitHub repository.
3. Create or edit a file named `README.md`.
4. Paste the copied code into the `README.md` file.
5. Save or commit the changes, and the README will be rendered on your repository's main page.

Let me know if you need any adjustments!
