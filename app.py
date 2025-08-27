
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Mantenimiento Predictivo Camanchaca", layout="wide")

st.title("⚙️ Mantenimiento Predictivo - Camanchaca")

uploaded_file = st.file_uploader("📂 Sube un archivo CSV con datos de sensores", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Vista previa de los datos cargados")
    st.dataframe(df.head())

    # Cargar modelo entrenado (ajusta el nombre al tuyo)
    try:
        modelo = joblib.load("modelo_rf.joblib")
    except:
        st.error("❌ No se encontró el archivo 'modelo_rf.joblib'. Asegúrate de guardarlo desde el script de entrenamiento.")
        st.stop()

    if st.button("🔮 Predecir fallas"):
        predicciones = modelo.predict(df.drop(columns=["falla"], errors="ignore"))
        st.success("✅ Resultados de la predicción")
        st.write(predicciones)
