import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from xgboost import XGBRegressor

final_df = pd.read_csv('t20internationaldf.csv')

def bowl(row):
    for team in eval(row['teams']) if isinstance(row['teams'], str) else row['teams']:
        if team != row['batting_team']:
            return team

final_df['bowling_team'] = final_df.apply(bowl, axis=1)
final_df.drop(columns=['teams'], inplace=True)

teams = [
    "India", "Australia", "England", "Pakistan", "South Africa",
    "New Zealand", "Sri Lanka", "Bangladesh", "West Indies",
    "Afghanistan", "Ireland", "Zimbabwe"
]
final_df = final_df[(final_df['batting_team'].isin(teams)) & (final_df['bowling_team'].isin(teams))]

final_df['wicket'] = np.where(final_df['player_dismissed'] == '0', 0, 1)
final_df.drop(columns=['player_dismissed'], inplace=True)
final_df['current_score'] = final_df.groupby('match_id')['runs'].cumsum()
final_df['over'] = final_df['ball'].apply(lambda x: str(x).split(".")[0]).astype(int)
final_df['ball_in_over'] = final_df['ball'].apply(lambda x: str(x).split(".")[1]).astype(int)
final_df['balls_done'] = final_df['over']*6 + final_df['ball_in_over']
final_df['balls_left'] = 120 - final_df['balls_done']
final_df['wickets_fallen'] = final_df.groupby('match_id')['wicket'].cumsum()
final_df['wickets_left'] = 10 - final_df['wickets_fallen']
final_df['crr'] = (final_df['current_score'] * 6) / final_df['balls_done']
final_df['last_5_overs_runs'] = final_df.groupby('match_id')['runs'].rolling(30, min_periods=1).sum().reset_index(0, drop=True)
final_df['last_over_runs'] = final_df.groupby('match_id')['runs'].rolling(6, min_periods=1).sum().reset_index(0, drop=True)
final_df['last_5_over_crr'] = final_df['last_5_overs_runs'] * 6 / np.minimum(final_df['balls_done'], 30)

win_df = final_df.copy()

# FIRST INNINGS SCORE PREDICTION
first_innings_df = final_df[final_df['inning'] == '1st innings'].copy()
final_scores = first_innings_df.groupby('match_id')['current_score'].max().reset_index()
final_scores.rename(columns={'current_score': 'final_score'}, inplace=True)
first_innings_df = first_innings_df.merge(final_scores, on='match_id')

x = first_innings_df.drop(columns=['final_score', 'venue', 'inning', 'ball', 'batsman', 'bowler', 'wicket', 'over', 'ball_in_over', 'balls_done'])
y = first_innings_df['final_score']

categorical_cols = ['batting_team', 'bowling_team', 'city']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
], remainder='passthrough')

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)

pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=10, random_state=42))
])
pipe.fit(X_train, Y_train)
y_predict = pipe.predict(X_test)

print("\nXGBoost:")
print("MAE:", mean_absolute_error(Y_test, y_predict))
print("R2:", r2_score(Y_test, y_predict))
accuracy = np.mean(np.abs(y_predict - Y_test) <= 10) * 100
print("Accuracy (±10 runs):", round(accuracy, 2), "%")
pickle.dump(pipe, open('internationalt20xgb.pkl', 'wb'))
