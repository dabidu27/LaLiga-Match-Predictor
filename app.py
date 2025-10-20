import streamlit as st
import pandas as pd
import joblib
from predictor import matches_df_rolling
from sqlalchemy import create_engine



db_url = st.secrets['database_url']
engine = create_engine(db_url)

def standardize_team_name(team_name):
    """Standardize team names to match historical data"""
    name_mapping = {
        'Betis': 'Real Betis',
        'Sociedad': 'Real Sociedad',
        'Athletic': 'Athletic Club',
        'Athletic Bilbao': 'Athletic Club',
        'Atlético Madrid': 'Atletico Madrid',
        'Rayo': 'Rayo Vallecano',
        'Villareal': 'Villarreal',  # Common spelling error
        'Alavés': 'Alaves',
        'Cádiz': 'Cadiz',
        # Add more mappings as needed
    }
    return name_mapping.get(team_name, team_name)

@st.cache_resource
def load_models():
    lr = joblib.load('lr_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return lr, scaler

@st.cache_data
def load_upcoming_matches():
    query = """SELECT * FROM upcoming_matches"""
    with engine.connect() as conn:
        df = pd.read_sql(query, con = engine)
    return df

st.set_page_config(page_title="LaLiga Match Predictor", layout="centered")
st.title("LaLiga Match Outcome Predictor")

st.markdown("""
This app predicts the outcome of upcoming La Liga matches using a Logistic Regression ML Model.
""")

lr, scaler = load_models()
upcoming_matches = load_upcoming_matches()

teams = sorted(upcoming_matches['Name'].unique())
team = st.selectbox("Select a team:", teams)

# ...existing code...

if team:
    st.subheader(f"Next match for {team}")
    team_matches = upcoming_matches[upcoming_matches['Name'] == team]

    if not team_matches.empty:
        next_team_match = team_matches.iloc[0]

        team = standardize_team_name(team)
        opp = standardize_team_name(next_team_match['Opponent'])
        
        match_display = pd.DataFrame({
            'Date': [next_team_match['Date'].strftime('%Y-%m-%d') if pd.notna(next_team_match['Date']) else 'TBD'],
            'Opponent': [str(opp)],
            'Venue': [str(next_team_match['Venue'])],
            'Time': [str(next_team_match['Time']) if pd.notna(next_team_match['Time']) else 'TBD'],
            'Round': [str(next_team_match['Round']) if pd.notna(next_team_match['Round']) else 'TBD']
        })

        st.write(match_display)

        matches_df_rolling['Date'] = pd.to_datetime(matches_df_rolling['Date'], errors='coerce')
        matches_df_rolling = matches_df_rolling.dropna(subset=['Date'])


        team_last = matches_df_rolling[matches_df_rolling['Name'] == team].sort_values('Date').iloc[-1]
        opp_last = matches_df_rolling[matches_df_rolling['Name'] == opp].sort_values('Date').iloc[-1]

        opponent_categories = matches_df_rolling['Opponent'].astype('category').cat.categories
        opponent_code = list(opponent_categories).index(opp) if opp in opponent_categories else -1

        next_match = pd.DataFrame([{
            'Venue_code': next_team_match['Venue_code'],
            'Opponent_code': opponent_code,
            'Hour': next_team_match['Hour'],
            'Day_code': next_team_match['Day_code'],
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

        next_match_scaled = scaler.transform(next_match)

        pred = lr.predict(next_match_scaled)[0]
        proba = lr.predict_proba(next_match_scaled)[0]

        st.markdown("### Model Prediction")
        if pred == 1:
            st.success(f"{team} is predicted to WIN with {proba[1]: .2%} of winning")
        else:
            st.warning(f"{team} likely to LOSE or DRAW! with {proba[1]: .2%} of winning")

    else:
        st.warning("No upcoming match found for this team.")
