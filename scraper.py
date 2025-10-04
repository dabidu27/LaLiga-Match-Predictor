from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup, Comment
import pandas as pd

URL = "https://fbref.com/en/comps/12/La-Liga-Stats"

#GET THE URL FOR EACH TEAM FROM THE STANDINGS TABLE

#use playwright instead of requests to launch a real browser in background, for webscraping to work
with sync_playwright() as p:

    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
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

#GET MATCH DATA FOR 1 TEAM
team_url = team_links[1]
with sync_playwright() as p:

    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto(team_url, timeout=60000)
    page.wait_for_selector('div#content')
    html = page.content()
    browser.close()

#get matches data for the team (drop columns where result = nan means drop matches that did not happen yet)
matches = pd.read_html(html, match='Scores & Fixtures')[0].dropna(subset='Result')

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
print(shooting.head())
print(matches.head())

team_data = matches.merge(shooting[['Date', 'Sh', 'SoT', 'Dist', 'FK', 'PK', 'PKatt']]  , on = 'Date') #merge matches and shooting data
print(team_data)