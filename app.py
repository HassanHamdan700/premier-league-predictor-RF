import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import datetime

st.set_page_config(page_title="PL Predictor Ultimate", layout="centered")

@st.cache_resource
def load_data():
    try:
        model = joblib.load('football_model_2526.pkl')
        
        with open('team_encoding_252.pkl', 'rb') as f:
            encodings = pickle.load(f)
            
        with open('team_stats_252.pkl', 'rb') as f:
            stats_df = pickle.load(f)
        
        stats_df['Date'] = pd.to_datetime(stats_df['Date'], dayfirst=True, errors='coerce')
        
        return model, encodings, stats_df
    except FileNotFoundError as e:
        st.error(f"Missing file: {e}. Please ensure all .pkl files are in the same directory.")
        return None, None, None

model, encodings, stats_df = load_data()

def find_match_in_history(home, away, target_date, df):
    """
    Searches for a match with a flexible date window (+/- 1 day)
    """
    target_ts = pd.Timestamp(target_date)
    start_window = target_ts - pd.Timedelta(days=1)
    end_window = target_ts + pd.Timedelta(days=1)
    
    mask = (
        (df['HomeTeam'] == home) & 
        (df['AwayTeam'] == away) & 
        (df['Date'] >= start_window) & 
        (df['Date'] <= end_window)
    )
    
    match_row = df[mask]
    
    if not match_row.empty:
        return match_row.iloc[0] 
    return None

st.title("Premier League Predictor")
st.markdown("Predict outcomes using Random Forest (2015-2025)")

if model is not None:
    
    teams = encodings['HomeTeam']
    
    default_home = list(teams).index("Liverpool") if "Liverpool" in teams else 0
    default_away = list(teams).index("Nott'm Forest") if "Nott'm Forest" in teams else 1
    
    col1, col2 = st.columns(2)
    home_team = col1.selectbox("Home Team", teams, index=default_home)
    away_team = col2.selectbox("Away Team", teams, index=default_away)
    
    col3, col4 = st.columns(2)
    date_input = col3.date_input("Match Date", datetime.date(2025, 11, 22))
    time_input = col4.time_input("Match Time", datetime.time(15, 0))

    historical_match = find_match_in_history(home_team, away_team, date_input, stats_df)
    is_historical = historical_match is not None

    if is_historical:
        st.success(f"Historical match data found for {home_team} vs {away_team}!")
    else:
        st.warning(f"Match not found in history. Using fallback stats.")

    st.divider()

    st.subheader("Betting Odds")
    
    use_exact_odds = False
    if is_historical:
        use_exact_odds = st.checkbox("Use exact saved odds from dataset?", value=True)
    
    if use_exact_odds and is_historical:
        prob_h = historical_match['prob_H']
        prob_d = historical_match['prob_D']
        prob_a = historical_match['prob_A']
        st.info(f"Using Historical Odds: Home {1/prob_h:.2f} | Draw {1/prob_d:.2f} | Away {1/prob_a:.2f}")
    else:
        c1, c2, c3 = st.columns(3)
        odds_h = c1.number_input("Home Win Odds", value=1.44, step=0.01)
        odds_d = c2.number_input("Draw Odds", value=5.00, step=0.1)
        odds_a = c3.number_input("Away Win Odds", value=6.25, step=0.1)
        
        prob_h = 1 / odds_h
        prob_d = 1 / odds_d
        prob_a = 1 / odds_a

    if st.button("Predict Result", type="primary"):
        if home_team == away_team:
            st.error("Home and Away teams cannot be the same.")
        else:
            match_datetime = pd.to_datetime(f"{date_input} {time_input}")
            hour = match_datetime.hour
            day_code = match_datetime.dayofweek
            
            home_code = list(teams).index(home_team)
            away_code = list(teams).index(away_team)
            
            if is_historical:
                stats = {
                    "FTHG_rolling": historical_match["FTHG_rolling"],
                    "FTAG_rolling": historical_match["FTAG_rolling"],
                    "HS_rolling": historical_match["HS_rolling"],
                    "AS_rolling": historical_match["AS_rolling"],
                    "HST_rolling": historical_match["HST_rolling"],
                    "AST_rolling": historical_match["AST_rolling"],
                    "Away_FTHG_rolling": historical_match["Away_FTHG_rolling"],
                    "Away_FTAG_rolling": historical_match["Away_FTAG_rolling"],
                    "Away_HS_rolling": historical_match["Away_HS_rolling"],
                    "Away_AS_rolling": historical_match["Away_AS_rolling"],
                    "Away_HST_rolling": historical_match["Away_HST_rolling"],
                    "Away_AST_rolling": historical_match["Away_AST_rolling"],
                }
            else:
                last_home = stats_df[(stats_df['HomeTeam'] == home_team) & (stats_df['Date'] < match_datetime)].sort_values('Date').iloc[-1:]
                last_away = stats_df[(stats_df['AwayTeam'] == away_team) & (stats_df['Date'] < match_datetime)].sort_values('Date').iloc[-1:]
                
                def get_val(df, col): return df.iloc[0][col] if not df.empty else 0

                stats = {
                    "FTHG_rolling": get_val(last_home, "FTHG_rolling"),
                    "FTAG_rolling": get_val(last_home, "FTAG_rolling"),
                    "HS_rolling": get_val(last_home, "HS_rolling"),
                    "AS_rolling": get_val(last_home, "AS_rolling"),
                    "HST_rolling": get_val(last_home, "HST_rolling"),
                    "AST_rolling": get_val(last_home, "AST_rolling"),
                    "Away_FTHG_rolling": get_val(last_away, "Away_FTHG_rolling"),
                    "Away_FTAG_rolling": get_val(last_away, "Away_FTAG_rolling"),
                    "Away_HS_rolling": get_val(last_away, "Away_HS_rolling"),
                    "Away_AS_rolling": get_val(last_away, "Away_AS_rolling"),
                    "Away_HST_rolling": get_val(last_away, "Away_HST_rolling"),
                    "Away_AST_rolling": get_val(last_away, "Away_AST_rolling"),
                }

            input_row = pd.DataFrame([{
                "home_code": home_code,
                "away_code": away_code,
                "hour": hour,
                "day_code": day_code,
                **stats,
                "prob_H": prob_h,
                "prob_D": prob_d,
                "prob_A": prob_a
            }])

            prediction = model.predict(input_row)[0]
            probs = model.predict_proba(input_row)[0]
            
            mapping = {0: "Away Win", 1: "Draw", 2: "Home Win"}
            result_text = mapping[prediction]
            
            if prediction == 2: color = "green"
            elif prediction == 0: color = "red" 
            else: color = "grey"
            
            st.markdown(f"<h2 style='text-align: center; color: {color}'>Prediction: {result_text}</h2>", unsafe_allow_html=True)
            
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Home Win", f"{probs[2]*100:.1f}%")
            col_b.metric("Draw", f"{probs[1]*100:.1f}%")
            col_c.metric("Away Win", f"{probs[0]*100:.1f}%")

    st.divider()
    st.subheader("Data Inspector (Debug Tool)")
    st.write("Check if your file actually contains the 2025 match data.")

    inspect_team = st.selectbox("Select Team to Inspect:", teams, key="debug_team")
    
    debug_data = stats_df[stats_df['HomeTeam'] == inspect_team].sort_values('Date', ascending=False)
    
    st.write(f"Top 5 most recent matches for **{inspect_team}** in your file:")
    st.dataframe(debug_data[['Date', 'HomeTeam', 'AwayTeam', 'FTR']].head(5))
    
    st.info("Check the dates above. If the latest date is 2024 (or earlier), your pickle file is outdated and needs to be regenerated in Colab.")
