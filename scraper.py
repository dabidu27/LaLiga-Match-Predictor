from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup, Comment
import pandas as pd
import time
from tqdm import tqdm
import random
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()

db_url = os.getenv('database_url')
db_engine = create_engine(db_url)

URL = "https://fbref.com/en/comps/12/La-Liga-Stats"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:115.0) Gecko/20100101 Firefox/115.0"
]

#GET THE URL FOR EACH TEAM FROM THE STANDINGS TABLE
def get_team_links(URL):
    #use playwright instead of requests to launch a real browser in background, for webscraping to work
    with sync_playwright() as p:

        browser = p.chromium.launch(headless=True)
        
        page = browser.new_page(user_agent=random.choice(USER_AGENTS))

        page.goto(URL, timeout = 60000)
        page.wait_for_selector('div#content')
        html = page.content()
        browser.close()

    #beautify html
    soup = BeautifulSoup(html, 'html.parser')

    standings_table = soup.select('table.stats_table')[0] #select the standings table
    links = standings_table.find_all('a') #find all the a tags in the standings table
    links = [l.get('href') for l in links] #get the links from the standings table (from href which is next to the a tag)
    links = [l for l in links if 'squads' in l] #get only the links for the teams
    team_links = ['https://fbref.com'+l for l in links] #create absolute link
    
    return team_links

#CREATE A FUNCTION TO GET MATCH DATA FOR 1 TEAM

def get_team_data(team_url):
    with sync_playwright() as p:

        browser = p.chromium.launch(headless=True)

        page = browser.new_page(user_agent=random.choice(USER_AGENTS))

        page.goto(team_url, timeout=60000)
        page.wait_for_selector('div#content')
        html = page.content()
        browser.close()

    #get matches data for the team (drop columns where result = nan means drop matches that did not happen yet)

    matches = pd.read_html(html, match='Scores & Fixtures')[0]
    matches = matches.dropna(subset=['Result'])


    #GET SHOOTING DATA FOR 1 TEAM (shooting data tells what happens in a match)
    soup = BeautifulSoup(html, 'html.parser') #beautify html
    links = soup.find_all('a') #get all a tags from the page
    links = [l.get('href') for l in links] #get all links from the page
    links = [l for l in links if l and 'all_comps/shooting' in l] #get the link for the shooting page

    #go to the shooting page
    with sync_playwright() as p:

        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(f'https://fbref.com{links[0]}', timeout=60000)
        page.wait_for_selector('div#content')
        html = page.content()
        browser.close()

    shooting = pd.read_html(html, match='Shooting')[0].dropna() #read the shooting stats table
    shooting.columns = shooting.columns.droplevel() #drop the overhead index
    #print(shooting.head())
    #print(matches.head())
    
    team_data = matches.merge(shooting[['Date', 'Sh', 'SoT', 'Dist', 'FK', 'PK', 'PKatt']]  , on = 'Date') #merge matches and shooting data
    team_name = team_url.split('/')[-1].replace('-Stats', "").replace('-', " ")
    team_data['Name'] = team_name
    team_data = team_data[team_data['Comp'] == 'La Liga']
    
    return team_data

#team_links = get_team_links(URL)
#team_data = get_team_data(team_links[0])
#print(team_data.columns)


#USE PREVIOUS FUNCTION TO SCRAPE DATA ALL TEAMS FOR MULTIPLE SEASONS


def scrape_all_seasons(start_year = 2025, end_year = 2022):

    all_seasons_data = []
    
    URL = "https://fbref.com/en/comps/12/La-Liga-Stats" #2025 - 2026 season

    for year in tqdm(range(start_year, end_year - 1, -1), desc="Scraping seasons"): #loop will stop when we reach 2018-2019 season

        team_links = get_team_links(URL) #get all the team links from the season we are currently scraping

        print(f"Scraping season page: {URL}")
        print(f"Found {len(team_links)} teams")


        for team in tqdm(team_links, desc=f"Teams in {year}-{year+1}", leave=False, ncols = 90): #for each team link
            
            try:
                team_data = get_team_data(team) #get the merged match and shooting data
                print(f"  → Scraping team: {team}")
            except ValueError:
                print(f"  ⚠️  Skipped team (no data found): {team}")
                continue

            team_data['Season'] = year
            all_seasons_data.append(team_data)
            time.sleep(random.uniform(3, 6))
        
        #find previous season button link
        with sync_playwright() as p:

            browser = p.chromium.launch(headless=True)
            page = browser.new_page(user_agent=random.choice(USER_AGENTS))
            page.goto(URL, timeout=60000)
            page.wait_for_selector('div#content')
            html = page.content()
            browser.close()
        
        soup = BeautifulSoup(html, 'html.parser')
        previous_season = soup.select_one('a.prev')
        if previous_season:
            URL = "https://fbref.com" + previous_season.get('href')

    print(f"✅ Finished. Collected {len(all_seasons_data)} DataFrames.")
    return pd.concat(all_seasons_data, ignore_index=True)

laliga_data = scrape_all_seasons()
print(laliga_data)

with db_engine.connect() as conn:
    laliga_data.to_sql("matches", con=db_engine, if_exists='replace', index = False)
        
print("Data successfully uploaded to PostgreSQL")