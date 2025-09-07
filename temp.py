import streamlit as st
import pandas as pd
import pickle

# ===============================
# Load Pretrained Models
# ===============================
pipeipl = pickle.load(open("iplscore.pkl", "rb"))
pipet20 = pickle.load(open("internationalt20xgb.pkl", "rb"))
pipeodi = pickle.load(open("odiinternationalt20xgb.pkl", "rb"))

# ===============================
# Predefined Options
# ===============================
teams_list = [
    "Australia", "India", "England", "Pakistan", "South Africa",
    "New Zealand", "Sri Lanka", "West Indies", "Bangladesh", "Afghanistan"
]

venues_list = [
    "Melbourne Cricket Ground", "Eden Gardens", "Lord's", "Wankhede Stadium",
    "SCG", "Adelaide Oval", "The Oval", "Dubai International Stadium",
    "Sharjah Cricket Stadium", "Gaddafi Stadium"
]

# ===============================
# Prediction Function
# ===============================
def predict_score(input_df, match_type):
    if match_type.lower() == "ipl":
        pred = pipeipl.predict(input_df)[0]
    elif match_type.lower() == "t20":
        pred = pipet20.predict(input_df)[0]
    else:  # ODI
        pred = pipeodi.predict(input_df)[0]
    return pred

# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="Manual Cricket Predictor", page_icon="🏏", layout="wide")
st.title("🏏 Manual Cricket Score Predictor (ODI, T20, IPL)")

st.sidebar.header("Enter Match Details")

# Dropdowns for match type, teams, and venue
match_type = st.sidebar.selectbox("Match Type", ["IPL", "T20", "ODI"])
batting_team = st.sidebar.selectbox("Batting Team", teams_list)
bowling_team = st.sidebar.selectbox("Bowling Team", [t for t in teams_list if t != batting_team])
venue = st.sidebar.selectbox("Venue / City", venues_list)

# Numeric inputs
current_score = st.sidebar.number_input("Current Score", min_value=0, value=0)
wickets_down = st.sidebar.number_input("Wickets Lost", min_value=0, max_value=10, value=0)
overs_done = st.sidebar.number_input("Overs Done (e.g., 12.3)", min_value=0.0, value=0.0, step=0.1)

# Calculate balls bowled correctly
overs_int = int(overs_done)
balls_fraction = int(round((overs_done - overs_int) * 10))  # .3 → 3 balls
balls_bowled = overs_int * 6 + balls_fraction

# Total balls based on match type
total_balls = 20 * 6 if match_type.lower() in ["t20", "ipl"] else 50 * 6
balls_left = max(total_balls - balls_bowled, 0)

# Correct CRR
overs_actual = balls_bowled / 6 if balls_bowled > 0 else 1
crr = current_score / overs_actual

st.sidebar.markdown(f"**Balls Bowled:** {balls_bowled}")
st.sidebar.markdown(f"**Balls Left:** {balls_left}")
st.sidebar.markdown(f"**Current Run Rate (CRR):** {crr:.2f}")

# Predict button
if st.sidebar.button("Predict Score"):
    input_df = pd.DataFrame([{
        "batting_team": batting_team,
        "bowling_team": bowling_team,
        "city": venue,
        "current_score": current_score,
        "balls_left": balls_left,
        "wickets_left": 10 - wickets_down,
        # Extra columns for T20/ODI models
        "runs": current_score,
        "wickets_fallen": wickets_down,
        "match_id": 0,
        "last_over_runs": 0,
        "last_5_overs_runs": 0,
        "last_5_over_crr": 0,
        "crr": crr
    }])

    predicted_score = predict_score(input_df, match_type)
    st.success(f"🏏 Predicted Final Score: {predicted_score:.0f}")
