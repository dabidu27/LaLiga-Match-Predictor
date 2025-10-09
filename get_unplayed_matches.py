from scraper import get_team_links
from playwright.sync_api import sync_playwright
import pandas as pd
import random
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()
db_url = os.getenv('database_url')
engine = create_engine(db_url)


USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:115.0) Gecko/20100101 Firefox/115.0"
]

def unplayed_matches():

    URL = "https://fbref.com/en/comps/12/La-Liga-Stats"
    team_links = get_team_links(URL)

    full_schedule = pd.DataFrame()
    with sync_playwright() as p:

        browser = p.chromium.launch(headless=True)

        page = browser.new_page(user_agent=random.choice(USER_AGENTS))

        for team in team_links:

            page.goto(team, timeout=60000)
            page.wait_for_selector('div#content')
            html = page.content()

            matches = pd.read_html(html, match='Scores & Fixtures')[0]
            team_name = team.split('/')[-1].replace('-Stats', "").replace('-', " ")
            matches['Name'] = team_name
            matches = matches[matches['Comp'] == 'La Liga']
            matches['Venue_code'] = matches['Venue'].astype('category').cat.codes

            if 'Time' in matches.columns:
                matches['Hour'] = matches['Time'].str.replace(':.*', '', regex=True)
                matches['Hour'] = pd.to_numeric(matches['Hour'], errors='coerce')
            else:
                matches['Hour'] = None

            matches['Date'] = pd.to_datetime(matches['Date'])
            matches['Day_code'] = matches['Date'].dt.dayofweek

            matches = matches[['Date', 'Time', 'Comp', 'Round', 'Day', 'Venue', 'Result', 'Name', 'Opponent', 'Venue_code', 'Hour', 'Day_code']]
            
            full_schedule = pd.concat([full_schedule, matches], ignore_index=True)

        browser.close()

    upcoming = full_schedule[full_schedule['Result'].isna()].copy()
    upcoming = upcoming.sort_values('Date', ascending=True).reset_index(drop=True)

    with engine.connect() as conn:

        upcoming.to_sql("upcoming_matches", con = engine, if_exists='replace', index=False)

unplayed_matches()
    







