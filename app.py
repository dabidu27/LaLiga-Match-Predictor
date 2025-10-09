import joblib
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
from predictor import matches_df_rolling

lr = joblib.load('lr_model.pkl')
scaler = joblib.load('scaler.pkl')

team = 'Real Madrid'
opp = 'Getafe'

team_last = matches_df_rolling[matches_df_rolling['Name'] == team].sort_values('Date').iloc[-1]
opp_last = matches_df_rolling[matches_df_rolling['Name'] == opp].sort_values('Date').iloc[-1]

# Use the same categorical mapping
opponent_categories = matches_df_rolling['Opponent'].astype('category').cat.categories

# Encode the upcoming opponent properly
opponent_code = list(opponent_categories).index('Getafe') if 'Getafe' in opponent_categories else -1

print(team_last)
print(opp_last)

next_match = pd.DataFrame([{
    'Venue_code': 0,  
    'Opponent_code': opponent_code,
    'Hour': 22,       
    'Day_code': 6,  
    'GF_roll': team_last['GF_roll'],
    'GA_roll': team_last['GA_roll'],
    'Sh_roll': team_last['Sh_roll'],
    'SoT_roll': team_last['SoT_roll'],
    'Dist_roll': team_last['Dist_roll'],
    'FK_roll': team_last['FK_roll'],
    'PK_roll': team_last['PK_roll'],
    'PKatt_roll': team_last['PKatt_roll'],
    'Points_roll': team_last['Points_roll'],
    'Opp_GF_roll': opp_last['GF_roll'],
    'Opp_GA_roll': opp_last['GA_roll'],
    'Opp_Sh_roll': opp_last['Sh_roll'],
    'Opp_SoT_roll': opp_last['SoT_roll'],
    'Opp_Dist_roll': opp_last['Dist_roll'],
    'Opp_FK_roll': opp_last['FK_roll'],
    'Opp_PK_roll': opp_last['PK_roll'],
    'Opp_PKatt_roll': opp_last['PKatt_roll'],
    'Opp_Points_roll': opp_last['Points_roll']
}])


pred = lr.predict(next_match)[0]
proba = lr.predict_proba(next_match)[0]

print('Prediction: ', pred)
print('Probabilities: ', proba)


