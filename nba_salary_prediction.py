import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import os

sns.set_theme(style="whitegrid")
filename = 'NBA Player Stats and Salaries_2010-2025.csv'

if not os.path.exists(filename):
    print(f"Error: {filename} not found.")
else:
    df = pd.read_csv(filename)

    # Data Cleaning
    target_col = 'Salary' if 'Salary' in df.columns else 'salary'
    if target_col in df.columns and df[target_col].dtype == 'O':
        df[target_col] = df[target_col].str.replace('$', '', regex=False).str.replace(',', '', regex=False).astype(float)

    df = df.fillna(0)

    le = LabelEncoder()
    team_col = 'Team' if 'Team' in df.columns else ('Tm' if 'Tm' in df.columns else None)
    if team_col:
        df['Team_Encoded'] = le.fit_transform(df[team_col].astype(str))

    # EDA: Salary Distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(df[target_col], kde=True, color='green', bins=30)
    plt.title('NBA Salary Distribution')
    plt.ticklabel_format(style='plain', axis='x')
    plt.show()

    # Dynamic Feature Selection
    features = ['Age', 'GP', 'G', 'MP', 'MIN', 'PTS', 'TRB', 'AST', 'STL', 'BLK', 'FG%', '3P%', 'Team_Encoded']
    valid_features = [col for col in features if col in df.columns]

    X = df[valid_features]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Training
    model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8, random_state=42)
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
    print(f"RMSE: ${np.sqrt(mean_squared_error(y_test, y_pred)):,.2f}")

    # Visualizations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Actual vs Predicted
    ax1.scatter(y_test, y_pred, alpha=0.5, color='purple')
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax1.set_title('Actual vs Predicted Salaries')
    ax1.ticklabel_format(style='plain', axis='both')

    # Feature Importance (Warning fixed here)
    importances = model.feature_importances_
    indices = np.argsort(importances)
    sns.barplot(
        x=importances[indices],
        y=[valid_features[i] for i in indices],
        ax=ax2,
        palette='viridis',
        hue=[valid_features[i] for i in indices],
        legend=False
    )
    ax2.set_title('Key Factors Affecting Salary')

    plt.tight_layout()
    plt.show()

    # Prediction Test
    test_data = pd.DataFrame({
        'Age': [28, 22], 'GP': [78, 30], 'G': [78, 30], 'MP': [34.5, 12.0],
        'MIN': [34.5, 12.0], 'PTS': [26.5, 5.2], 'TRB': [8.4, 2.1], 'AST': [6.2, 1.0],
        'STL': [1.5, 0.3], 'BLK': [0.8, 0.1], 'FG%': [0.495, 0.410], '3P%': [0.380, 0.290],
        'Team_Encoded': [5, 5]
    })[valid_features]

    preds = model.predict(test_data)
    print(f"\nStar Player Prediction: ${preds[0]:,.2f}")
    print(f"Bench Player Prediction: ${preds[1]:,.2f}")