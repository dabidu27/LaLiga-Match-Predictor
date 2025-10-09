from playwright.sync_api import sync_playwright
import psycopg2
import os
from dotenv import load_dotenv
from scraper import get_team_links, get_team_data
from sqlalchemy import create_engine
import pandas as pd

load_dotenv()
db_url_psy = os.getenv("database_url").replace('+psycopg2', '')


with psycopg2.connect(db_url_psy) as conn:

    cursor = conn.cursor()
    cursor.execute('SELECT "Date" FROM matches ORDER BY "Date" DESC LIMIT 1;')
    latest_date = cursor.fetchone()[0]

URL = 'https://fbref.com/en/comps/12/La-Liga-Stats'
team_links = get_team_links(URL)

db_url = os.getenv('database_url')
db_engine = create_engine(db_url)

def scrape_new_matches(team_links):

    for team in team_links:

        print(f"Scraping {team.split('/')[-1].replace('-Stats', '').replace('-', ' ')}")
        team_data = get_team_data(team)
        team_data['Date'] = pd.to_datetime(team_data['Date'], errors = 'coerce')

        if 'Attendance' in team_data.columns:
            team_data['Attendance'] = (
                team_data['Attendance']
                .replace({',': ''}, regex=True)
                .replace('-', None)
            )
            team_data['Attendance'] = pd.to_numeric(team_data['Attendance'], errors='coerce').astype('Int64')
            
        new_matches = team_data[(team_data['Date'] > latest_date) & (team_data['Result'].notna())]
        upcoming_matches = team_data[(team_data['Date'] > latest_date) & (team_data['Result'].isna())]
        if not new_matches.empty:

            new_matches.to_sql("matches", con = db_engine, if_exists='append', index=False)
            with psycopg2.connect(db_url_psy) as conn:

                cursor = conn.cursor()
                for _, row in new_matches.iterrows():
                    cursor.execute("""
                        DELETE FROM upcoming_matches
                        WHERE "Name" = %s AND "Date" = %s
                        """, (row['Name'], row['Date']))
                
                conn.commit()
                    
            print('Added new matches')
        else:
            print(f"No new matches found for {team_data['Name'].iloc[0]}")
            print('/n')


scrape_new_matches(team_links)