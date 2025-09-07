import streamlit as st
import requests
import pandas as pd
import pickle
import time

# ===============================
# Load Pretrained Models
# ===============================
pipeipl = pickle.load(open("iplscore.pkl", "rb"))
pipet20 = pickle.load(open("internationalt20xgb.pkl", "rb"))
pipeodi = pickle.load(open("odiinternationalt20xgb.pkl", "rb"))

# ===============================
# API Config
# ===============================
API_KEY = "fa1b13a7-01eb-4bac-946f-141920965621"   # Replace with CricketData.org API key
BASE_URL = "https://api.cricapi.com/v1/"

# ===============================
# Functions
# ===============================
def get_live_matches():
    url = f"{BASE_URL}currentMatches?apikey={API_KEY}&offset=0"
    response = requests.get(url).json()
    if "data" in response:
        return response["data"]
    return []

def parse_match_data(matches):
    rows = []
    for match in matches:
        match_type = match.get("matchType", "").lower()
        series_name = match.get("series", "").lower()
        if match_type in ["odi", "t20"] or "ipl" in series_name:
            if match.get("status") == "live":
                score_info = match.get("score", [{}])[0]
                batting_team = score_info.get("inning", "").split(" ")[0]
                teams = match.get("teams", [])
                bowling_team = teams[1] if teams and teams[0] == batting_team else teams[0]
                overs = score_info.get("o", 0)
                try:
                    overs = float(overs)
                    balls_done = int(overs) * 6 + int(round((overs - int(overs)) * 10))
                except:
                    balls_done = 0
                balls_left = 120 - balls_done if match_type == "t20" else 300 - balls_done
                info = {
                    "Match": match.get("name"),
                    "Type": match.get("matchType"),
                    "Venue": match.get("venue"),
                    "batting_team": batting_team,
                    "bowling_team": bowling_team,
                    "city": match.get("venue", "Unknown"),
                    "current_score": score_info.get("r", 0),
                    "wickets_left": 10 - score_info.get("w", 0),
                    "balls_left": balls_left,
                    "crr": (score_info.get("r", 0) * 6 / balls_done) if balls_done > 0 else 0,
                    "Status": match.get("status"),
                }
                rows.append(info)
    return pd.DataFrame(rows)

def predict_score(df):
    if df.empty:
        return df
    
    preds = []
    for _, row in df.iterrows():
        input_df = pd.DataFrame([{
            "batting_team": row["batting_team"],
            "bowling_team": row["bowling_team"],
            "city": row["city"],
            "current_score": row["current_score"],
            "balls_left": row["balls_left"],
            "wickets_left": row["wickets_left"],
            "crr": row["crr"],
        }])

        if "ipl" in row["Match"].lower():
            pred = pipeipl.predict(input_df)[0]
        elif row["Type"].lower() == "t20":
            pred = pipet20.predict(input_df)[0]
        else:  # ODI
            pred = pipeodi.predict(input_df)[0]

        preds.append(pred)

    df["Predicted_Score"] = preds
    return df

# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="Live Cricket Predictor", page_icon="🏏", layout="wide")
st.title("🏏 Live ODI, T20 & IPL Score Predictor")
placeholder = st.empty()

while True:
    matches = get_live_matches()
    df = parse_match_data(matches)
    df = predict_score(df)

    with placeholder.container():
        if not df.empty:
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("No live ODI, T20, or IPL matches right now.")

    time.sleep(3600)  # Auto-refresh every 30 sec
