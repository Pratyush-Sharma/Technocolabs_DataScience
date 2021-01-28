
import streamlit as st
import pandas as pd
import pickle
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Spotify Skip Prediction App
This app predicts whether the song will be **skipped** or not!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    duration = st.sidebar.slider('Duration', 30, 1800, 180)
    release_year = st.sidebar.slider('Release year', 1950, 2020, 1990)
    us_popularity_estimate = st.sidebar.slider('US popularity estimate', 90.0, 100.0, 92.0)
    session_comp = st.sidebar.slider('Session complete', 0.05, 1.00, 0.5)
    context_switch = st.sidebar.slider('Context switch', 0, 1, 0)
    no_pause_before_play = st.sidebar.slider('No pause before play', 0, 1, 0)
    pause_before_play = st.sidebar.slider('Pause before play', 0, 1, 0)
    hist_user_behavior_n_seekfwd = st.sidebar.slider('Seekforward', 0, 60, 1)
    hist_user_behavior_n_seekback = st.sidebar.slider('Seekback', 0, 151, 0)
    hist_user_behavior_is_shuffle = st.sidebar.slider('Shuffle', 0, 1, 0)
    Start_reason_appload = st.sidebar.slider('Start reason - appload', 0, 1, 0)
    Start_reason_backbtn = st.sidebar.slider('Start reason - back button', 0, 1, 0)
    Start_reason_clickrow = st.sidebar.slider('Start reason - clickrow', 0, 1, 0)
    Start_reason_fwdbtn = st.sidebar.slider('Start reason - forward button', 0, 1, 0)
    Start_reason_remote = st.sidebar.slider('Start reason - remote', 0, 1, 0)
    Start_reason_trackdone = st.sidebar.slider('Start reason - trackdone', 0, 1, 0)
    End_reason_backbtn = st.sidebar.slider('End reason - back button', 0, 1, 0)
    End_reason_endplay = st.sidebar.slider('End reason - end play', 0, 1, 0)
    End_reason_fwdbtn = st.sidebar.slider('End reason - forward button', 0, 1, 0)
    End_reason_logout = st.sidebar.slider('End reason - logout', 0, 1, 0)
    End_reason_remote = st.sidebar.slider('End reason - remote', 0, 1, 0)
    End_reason_trackdone = st.sidebar.slider('End reason - trackdone', 0, 1, 0)
    acousticness = st.sidebar.slider('Acousticness', 0.00, 1.00, 0.05)
    bounciness = st.sidebar.slider('Bounciness', 0.00, 1.00, 0.05)
    energy = st.sidebar.slider('Energy', 0.00, 1.00, 0.05)
    instrumentalness = st.sidebar.slider('Instrumentalness', 0.00, 1.00, 0.00)
    liveness = st.sidebar.slider('Liveness', 0.00, 1.00, 0.05)
    loudness = st.sidebar.slider('Loudness', -60.00, 10.00, -7.00)
    mechanism = st.sidebar.slider('Mechanism', 0.00, 1.00, 0.05)
    organism = st.sidebar.slider('Organism', 0.00, 1.00, 0.05)
    speechiness = st.sidebar.slider('Speechiness', 0.00, 1.00, 0.05)
    time_signature = st.sidebar.slider('Time signature', 0, 5, 4)
    valence = st.sidebar.slider('Valence', 0.00, 1.00, 0.05)
    acoustic_vector_0 = st.sidebar.slider('Acoustic vector 0', -1.20, 1.20, -0.50)
    acoustic_vector_1 = st.sidebar.slider('Acoustic vector 1', -1.20, 1.20, 0.20)
    acoustic_vector_2 = st.sidebar.slider('Acoustic vector 2', -1.20, 1.20, 0.20)
    acoustic_vector_3 = st.sidebar.slider('Acoustic vector 3', -1.20, 1.20, 0.00)
    acoustic_vector_4 = st.sidebar.slider('Acoustic vector 4', -1.20, 1.20, -0.10)
    acoustic_vector_5 = st.sidebar.slider('Acoustic vector 5', -1.20, 1.20, -0.05)
    acoustic_vector_6 = st.sidebar.slider('Acoustic vector 6', -1.20, 1.20, -0.20)
    acoustic_vector_7 = st.sidebar.slider('Acoustic vector 7', -1.20, 1.20, 0.00)

    data = {'Duration': duration,
            'Release year': release_year,
            'US popularity estimate': us_popularity_estimate,
            'Session complete': session_comp,
            'Context switch': context_switch,
            'No pause before play': no_pause_before_play,
            'Pause before play': pause_before_play,
            'Seekforward': hist_user_behavior_n_seekfwd,
            'Seekback': hist_user_behavior_n_seekback,
            'Shuffle': hist_user_behavior_is_shuffle,
            'Start reason - appload': Start_reason_appload,
            'Start reason - back button': Start_reason_backbtn,
            'Start reason - clickrow': Start_reason_clickrow,
            'Start reason - forward button': Start_reason_fwdbtn,
            'Start reason - remote': Start_reason_remote,
            'Start reason - trackdone': Start_reason_trackdone,
            'End reason - back button': End_reason_backbtn,
            'End reason - end play': End_reason_endplay,
            'End reason - forward button': End_reason_fwdbtn,
            'End reason - logout': End_reason_logout,
            'End reason - remote': End_reason_remote,
            'End reason - trackdone': End_reason_trackdone,
            'Acousticness': acousticness,
            'Bounciness': bounciness,
            'Energy ': energy,
            'Instrumentalness': instrumentalness,
            'Liveness': liveness,
            'Loudness': loudness,
            'Mechanism': mechanism,
            'Organism': organism,
            'Speechiness': speechiness,
            'Time signature': time_signature,
            'Valence': valence,
            'Acoustic vector 0': acoustic_vector_0,
            'Acoustic vector 1': acoustic_vector_1,
            'Acoustic vector 2': acoustic_vector_2,
            'Acoustic vector 3': acoustic_vector_3,
            'Acoustic vector 4': acoustic_vector_4,
            'Acoustic vector 5': acoustic_vector_5,
            'Acoustic vector 6': acoustic_vector_6,
            'Acoustic vector 7': acoustic_vector_7}


    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

#iris = datasets.load_iris()
#X = iris.data
#Y = iris.target

filename1 = "rf_model.sav"
rf_model = pickle.load(open(filename1, 'rb'))

#clf = RandomForestClassifier()
#clf.fit(X, Y)

prediction = rf_model.predict(df)
#prediction_proba = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
#st.write(iris.target_names)
st.write("""
| Prediction    | Label         |
| ------------- |:-------------:|
| 0             | Not Skipped   |
| 1             | Skipped       |
""")

st.subheader('Prediction')
st.write(prediction)
#st.write(prediction)

#st.subheader('Prediction Probability')
#st.write(prediction_proba)