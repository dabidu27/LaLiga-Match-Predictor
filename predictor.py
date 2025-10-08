from dotenv import load_dotenv
import os
from sqlalchemy import create_engine
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


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