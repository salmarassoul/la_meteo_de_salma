import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Configuration de la page
st.set_page_config(
    page_title="La Météo",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Style CSS personnalisé
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .weather-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 25px;
        border-radius: 18px;
        color: white;
        text-align: center;
        box-shadow: 0px 8px 16px rgba(0,0,0,0.15);
        margin-bottom: 25px;
    }
    .weather-header h1 {
        color: white !important;
        font-size: 2.3rem;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .metric-card {
        background-color: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.4);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        text-align: center;
        margin-bottom: 15px;
    }
    .metric-card h4 { color: #555; font-size: 0.95rem; margin-bottom: 8px; font-weight: 600; }
    .metric-value { font-size: 1.8rem; font-weight: bold; color: #1e3c72; }
    .result-card-rain {
        background: linear-gradient(135deg, #ff4e50 0%, #f9d423 100%);
        color: white; padding: 25px; border-radius: 18px; text-align: center;
    }
    .result-card-dry {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%);
        color: white; padding: 25px; border-radius: 18px; text-align: center;
    }
    .result-card-amount {
        background: linear-gradient(135deg, #2193b0 0%, #6dd5ed 100%);
        color: white; padding: 25px; border-radius: 18px; text-align: center;
    }
    .stButton>button {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white !important; border: none; border-radius: 12px;
        padding: 12px 28px; font-size: 1.1rem; font-weight: 600; width: 100%;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Header de l'application
st.markdown(
    """
    <div class="weather-header">
        <h1>🌤️ La Météo de Salma</h1>
        <p>Prédiction Météorologique par Machine Learning</p>
    </div>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_resources():
    model_clf = joblib.load("model_classification.pkl")
    model_reg = joblib.load("model_regression.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("model_columns.pkl")
    return model_clf, model_reg, scaler, columns


try:
    model_clf, model_reg, scaler, columns = load_resources()
except Exception as e:
    st.error(f"Erreur lors du chargement des fichiers de modèle : {e}")
    st.stop()

st.sidebar.image("https://img.icons8.com/fluency/96/weather.png", width=70)
st.sidebar.title("⚙️ Paramètres Météo")


def user_input_features():
    st.sidebar.subheader("🌡️ Températures")
    Temps_Min = st.sidebar.number_input("Minima (°C)", value=12.0, step=0.5)
    Temp_Max = st.sidebar.number_input("Maxima (°C)", value=22.0, step=0.5)

    st.sidebar.subheader("💧 Humidité & Pluie")
    Precipitation = st.sidebar.number_input(
        "Pluie aujourd'hui (mm)", value=0.0, step=0.1
    )
    Humidite_9h = st.sidebar.slider("Humidité Matin - 9h (%)", 0, 100, 65)
    Humidite_15h = st.sidebar.slider(
        "Humidité Après-midi - 15h (%)", 0, 100, 45
    )

    st.sidebar.subheader("📊 Pression")
    Pression_15h = st.sidebar.number_input(
        "Pression à 15h (hPa)", value=1015.0, step=0.5
    )

    data = pd.DataFrame(0, index=[0], columns=columns)

    mapping = {
        "MinTemp": Temps_Min,
        "MaxTemp": Temp_Max,
        "Rainfall": Precipitation,
        "Humidity9am": Humidite_9h,
        "Humidity3pm": Humidite_15h,
        "Pressure3pm": Pression_15h,
    }

    for col, val in mapping.items():
        if col in data.columns:
            data[col] = val

    return (
        data,
        Temps_Min,
        Temp_Max,
        Precipitation,
        Humidite_9h,
        Humidite_15h,
        Pression_15h,
    )


input_df, t_min, t_max, precip, hum_9, hum_15, press = user_input_features()

st.subheader("📋 Observations Météo Saisies")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f'<div class="metric-card"><h4>🌡️ Températures</h4><div class="metric-value">{t_min}°C / {t_max}°C</div></div>',
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f'<div class="metric-card"><h4>💧 Humidité (15h)</h4><div class="metric-value">{hum_15}%</div></div>',
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        f'<div class="metric-card"><h4>🌧️ Pluie Journée</h4><div class="metric-value">{precip} mm</div></div>',
        unsafe_allow_html=True,
    )
with col4:
    st.markdown(
        f'<div class="metric-card"><h4>⏲️ Pression (15h)</h4><div class="metric-value">{press} hPa</div></div>',
        unsafe_allow_html=True,
    )

if st.button("🚀 Obtenir le Bulletin de Salma pour Demain"):
    pred_class = model_clf.predict(input_df)[0]
    proba_class = model_clf.predict_proba(input_df)[0][1]
    pred_amount = max(0.0, model_reg.predict(input_df)[0])

    st.markdown("---")
    st.subheader("🎯 Prédictions Météorologiques pour Demain")
    col_res1, col_res2 = st.columns(2)

    with col_res1:
        if pred_class == 1:
            st.markdown(
                f'<div class="result-card-rain"><h3>🌧️ Risque de Pluie Élevé</h3><h1 style="font-size: 3rem; margin: 10px 0;">{proba_class*100:.1f}%</h1><p>Probabilité de précipitation demain</p></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="result-card-dry"><h3>☀️ Temps Sec Prévu</h3><h1 style="font-size: 3rem; margin: 10px 0;">{(1-proba_class)*100:.1f}%</h1><p>Probabilité de temps ensoleillé/sec</p></div>',
                unsafe_allow_html=True,
            )

    with col_res2:
        st.markdown(
            f'<div class="result-card-amount"><h3>📏 Hauteur de Pluie Estimée</h3><h1 style="font-size: 3rem; margin: 10px 0;">{pred_amount:.2f} mm</h1><p>Volume moyen prévu par le modèle de Régression</p></div>',
            unsafe_allow_html=True,
        )
