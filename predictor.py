from dotenv import load_dotenv
import os
from sqlalchemy import create_engine
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib

    
load_dotenv()
db_url = os.getenv('database_url')
engine = create_engine(db_url)

with engine.connect() as conn:

    matches_df = pd.read_sql_table("matches", con = engine)

matches_df.columns = matches_df.columns.map(str)

#CLEANING DATA
print(matches_df.head())
print(matches_df.isna().sum())
matches_df = matches_df.drop(columns='Notes')
print(matches_df.isna().sum())
print(matches_df.info())


matches_df['Date'] = pd.to_datetime(matches_df['Date'], errors = 'coerce')

num_cols = ['GF', 'GA', 'Sh', 'SoT', 'Dist', 'FK', 'PK', 'PKatt', 'Poss', 'xG', 'xGA']
matches_df[num_cols] = matches_df[num_cols].apply(pd.to_numeric, errors = 'coerce')
print(matches_df.dtypes)

str_cols = matches_df.select_dtypes(include='object').columns
matches_df[str_cols] = matches_df[str_cols].apply(lambda x: x.str.strip())

print(matches_df.head())

print(matches_df.shape)
print(matches_df['Name'].value_counts())
print(matches_df['Round'].value_counts())

#FEATURE ENGINEERING

matches_df['Venue_code'] = matches_df['Venue'].astype('category').cat.codes
matches_df['Opponent_code'] = matches_df['Opponent'].astype('category').cat.codes
matches_df['Hour'] = matches_df['Time'].str.replace(':.+', '', regex=True).astype(int)
matches_df['Day_code'] = matches_df['Date'].dt.dayofweek
print(matches_df.head())

#INITIAL ML MODELS

matches_df['target'] = (matches_df['Result'] == 'W').astype(int) #1 for W, 0 for L/D
print(matches_df)

print(matches_df.dtypes)

features = ['Venue_code', 'Opponent_code', 'Hour', 'Day_code'] #excluded xG, xGA for the moment to stop the model from 'cheating'

X = matches_df[features]
X.columns = X.columns.astype(str)
y = matches_df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

"""
#Logistic Regression model
LR = LogisticRegression(max_iter=1000)
LR.fit(X_train_scaled, y_train)
y_pred = LR.predict(X_test_scaled)

print('Logistic Regression Evaluation: ')
print('Accuracy', accuracy_score(y_test, y_pred))
print('Confussion matrix', confusion_matrix(y_test, y_pred))
print('Classification report', classification_report(y_test, y_pred))

coef_importance = pd.DataFrame({'Features': features, 'Coefficient': LR.coef_[0]}).sort_values('Coefficient', ascending=False)
print(coef_importance)


#Random Forest Classifier

kf = KFold(n_splits = 5, shuffle = True, random_state = 42)
RF = RandomForestClassifier(random_state = 42)

param_grid = {'n_estimators': [50, 100, 200], 'min_samples_split': [2, 5, 10]}
RF_cv = GridSearchCV(RF, param_grid, cv=kf)
RF_cv.fit(X_train_scaled, y_train)

print('Best parameters: ', RF_cv.best_params_)
print('Best CV accuracy: ', RF_cv.best_score_)

best_RF = RF_cv.best_estimator_
y_pred = best_RF.predict(X_test_scaled)

print('RandomForest Classifier evaluation: ')
print('Accuracy: ', accuracy_score(y_test, y_pred))
print('Confussion matrix: ', confusion_matrix(y_test, y_pred))
print('Classification report', classification_report(y_test, y_pred))


#xgboost

xgb = XGBClassifier(random_state = 42, use_label_encoder = False, eval_metric = 'logloss')

params_grid = {'n_estimators': [100, 200, 300], 'max_depth': [3,5,7], 'learning_rate': [0.01, 0.1, 0.2], 'subsample': [0.8, 1.0]}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

xgb_cv = GridSearchCV(xgb, params_grid, cv=kf)

xgb_cv.fit(X_train_scaled, y_train)

print('Best parameters: ', xgb_cv.best_params_)
print('Best accuracy: ', xgb_cv.best_score_)

best_xgb = xgb_cv.best_estimator_

y_pred = best_xgb.predict(X_test_scaled)

print('XGBoost Evaluation: ')
print('Accuracy: ', accuracy_score(y_test, y_pred))
print('Confusion matrix: ', confusion_matrix(y_test, y_pred))
print('Classification report: ', classification_report(y_test, y_pred))
"""

#MORE FEATURE ENGINEERING
#points per game

points = {'W': 3, 'D': 1, 'L': 0}
matches_df['Points'] = matches_df['Result'].map(points)
print(matches_df.head())

#rolling averages

grouped_matches = matches_df.groupby('Name')
#group = grouped_matches.get_group('Real Madrid')
#print(group)

def rolling_averages(group, cols, new_col):

    group = group.sort_values('Date')
    rolling_stats = group[cols].rolling(5, closed = 'left').mean()
    group[new_col] = rolling_stats
    group = group.dropna(subset = new_col)
    return group


cols = ['GF', 'GA', 'Sh', 'SoT', 'Dist', 'FK', 'PK', 'PKatt', 'Points']
new_cols = [f"{c}_roll" for c in cols]

matches_df_rolling = matches_df.groupby('Name').apply(lambda x: rolling_averages(x, cols, new_cols))
matches_df_rolling = matches_df_rolling.droplevel('Name')
matches_df_rolling.index = range(matches_df_rolling.shape[0])

opponent_stats = matches_df_rolling[['Name', 'Date', 'GF_roll', 'GA_roll', 'Sh_roll', 'SoT_roll', 'Dist_roll', 'FK_roll', 'PK_roll', 'PKatt_roll', 'Points_roll']]
opponent_stats.rename(columns= {'Name': 'Opponent', 'GF_roll': 'Opp_GF_roll', 'GA_roll': 'Opp_GA_roll','Sh_roll': 'Opp_Sh_roll', 'SoT_roll': 'Opp_SoT_roll', 'Dist_roll': 'Opp_Dist_roll', 'FK_roll': 'Opp_FK_roll', 'PK_roll': 'Opp_PK_roll', 'PKatt_roll': 'Opp_PKatt_roll', 'Points_roll': 'Opp_Points_roll'}, inplace=True)

matches_df_rolling = matches_df_rolling.merge(opponent_stats, on = ['Opponent', 'Date'], how = 'left')
matches_df_rolling = matches_df_rolling.fillna(0)


#print(matches_df_rolling)

if __name__ == '__main__':

    #test models
    features = ['Venue_code', 'Opponent_code', 'Hour', 'Day_code', 'GF_roll', 'GA_roll', 'Sh_roll', 'SoT_roll', 'Dist_roll', 'FK_roll', 'PK_roll', 'PKatt_roll', 'Points_roll', 'Opp_GF_roll', 'Opp_GA_roll', 'Opp_Sh_roll', 'Opp_SoT_roll', 'Opp_Dist_roll', 'Opp_FK_roll', 'Opp_PK_roll', 'Opp_PKatt_roll', 'Opp_Points_roll']

    X = matches_df_rolling[features]
    X.columns = X.columns.astype(str)

    y = matches_df_rolling['target']

    #Logistic Regression
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    y_pred = lr.predict(X_test_scaled)

    print('Logistic Regression Evaluation: ')
    print('Accuracy: ', accuracy_score(y_test, y_pred))
    print('Confussion matrix: ', confusion_matrix(y_test, y_pred))
    print('Classification report: ', classification_report(y_test, y_pred))

    coef_importance = pd.DataFrame({'features': features, 'coefficient': lr.coef_[0]}).sort_values('coefficient', ascending=False)
    print(coef_importance)
    
    #Random Forest

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rf = RandomForestClassifier(random_state=42)

    param_grid = {'n_estimators': [50, 100, 200], 'min_samples_split': [2, 5, 10]}
    rf_cv = GridSearchCV(rf, param_grid=param_grid, cv = kf)

    rf_cv.fit(X_train, y_train)
    best_rf = rf_cv.best_estimator_
    y_pred = best_rf.predict(X_test)

    print('Random Forest Evaluation: ')
    print('Accuracy: ', accuracy_score(y_test, y_pred))

    #XGBoost

    xgb = XGBClassifier(random_state = 42, use_label_encoder = False, eval_metric = 'logloss')

    params_grid = {'n_estimators': [100, 200, 300], 'max_depth': [3,5,7], 'learning_rate': [0.01, 0.1, 0.2], 'subsample': [0.8, 1.0]}
    xgb_cv = GridSearchCV(xgb, params_grid, cv = kf)
    xgb_cv.fit(X_train, y_train)

    best_xgb = xgb_cv.best_estimator_
    y_pred = best_xgb.predict(X_test)

    print('XGBoost Evaluation: ')
    print('Accuracy: ', accuracy_score(y_test, y_pred))

    
    #After evaluation, Logistic Regression proved to be the best model

    X_scaled_full = scaler.fit_transform(X)


    matches_df_rolling["Predicted_result"] = lr.predict(X_scaled_full)
    matches_df_rolling["Predicted_win_prob"] = lr.predict_proba(X_scaled_full)[:, 1]

    print(matches_df_rolling[['Name', 'Opponent', 'Date', 'Result', 'Predicted_result', 'Predicted_win_prob']])


    joblib.dump(lr, 'lr_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    #elo rating

