from dotenv import load_dotenv
import os
from sqlalchemy import create_engine
import pandas as pd



load_dotenv()
db_url = os.getenv('database_url')
engine = create_engine(db_url)

with engine.connect() as conn:

    matches_df = pd.read_sql_table("matches", con = engine)

print(matches_df.head())
print(matches_df.isna().sum())
matches_df = matches_df.drop(columns='Notes')
print(matches_df.isna().sum())

matches_df['Date'] = pd.to_datetime(matches_df['Date'], errors = 'coerce')
num_cols = ['GF', 'GA', 'Sh', 'SoT', 'Dist', 'FK', 'PK', 'PKatt', 'Poss']
matches_df[num_cols] = matches_df[num_cols].apply(pd.to_numeric, errors = 'coerce')
print(matches_df.dtypes)


#FEATURE ENGINEERING
