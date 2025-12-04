import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import streamlit.components.v1 as components

# ---------------------------
# Configuración inicial
# ---------------------------
st.set_page_config(page_title="Predicción de Churn", layout="wide")

st.title("📊 Predicción de Abandono de Clientes (Churn)")
st.write("Aplicación de Machine Learning con XGBoost y explicación con SHAP.")

# ---------------------------
# Cargar modelo y scaler
# ---------------------------
try:
    model = joblib.load("models/xgb_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
except:
    st.error("❌ No se pudieron cargar los modelos. Verifica la carpeta 'models/'.")
    st.stop()

feature_names = model.get_booster().feature_names

# ---------------------------
# Sidebar - Inputs del Usuario
# ---------------------------
st.sidebar.header("🧑‍💼 Datos del Cliente")

edad = st.sidebar.slider("Edad", 18, 80, 35)
tenure = st.sidebar.slider("Meses como cliente", 0, 72, 12)
monthly_charges = st.sidebar.number_input("Pago Mensual", 0.0, 300.0, 70.0)
total_charges = st.sidebar.number_input("Pago Total", 0.0, 20000.0, 2000.0)

# Ajusta estos nombres según TU dataset real
contract_month_to_month = st.sidebar.selectbox("Contrato Mensual", [0, 1])
internet_fiber = st.sidebar.selectbox("Internet Fibra Óptica", [0, 1])

input_data = np.array([[edad, tenure, monthly_charges, total_charges,
                         contract_month_to_month, internet_fiber]])

input_df = pd.DataFrame(input_data, columns=feature_names)

# ---------------------------
# Escalado
# ---------------------------
input_scaled = scaler.transform(input_df)

# ---------------------------
# Predicción
# ---------------------------
if st.button("🔮 Predecir Churn"):
    prob = model.predict_proba(input_scaled)[0][1]
    pred = model.predict(input_scaled)[0]

    if pred == 1:
        st.error(f"⚠️ El cliente probablemente ABANDONARÁ (Probabilidad: {prob:.2f})")
    else:
        st.success(f"✅ El cliente probablemente NO abandonará (Probabilidad: {prob:.2f})")

    # ---------------------------
    # SHAP Explicación Individual
    # ---------------------------
    explainer = shap.Explainer(model)
    shap_values = explainer(input_df)

    shap_html = f"<head>{shap.getjs()}</head><body>"
    shap_html += shap.plots.waterfall(shap_values[0], show=False).html()
    shap_html += "</body>"

    components.html(shap_html, height=450)


