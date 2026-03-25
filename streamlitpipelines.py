import streamlit as st
import pandas as pd
from joblib import load
import numpy as np

# Cargar modelo entrenado de clasificación
modelo_cloro = load('XGBoost.joblib')

# Variables seleccionadas para el modelo
features_modelo = [
    "Flag_Menos2anios_Eleccion",
    "Establecimiento_Vigila_Calidad_Agua_SI",
    "ApoyoMuni_Control_Calidad_Agua_SI",
    "cuotapromedio_Cat",
    "Flag_Miembros_SI_SecundariaSuperior",
    "Registro_Cloro_Residual_SI",
    "Realizan_Limpieza_SitemaAgua_SI",
    "Apoyo_Infraestructura",
    "Salud_Financiera_Cuotas_Porc",
    "Apoyo_Tecnico"
]

# Valores para inputs categóricos
cuotapromedio_options = ["2", "2_5", "5_10", "10"]

st.set_page_config(layout="wide")
st.title("Modelo de Clasificación eXtreme Gradient Boosting")
st.markdown("### Predicción de existencia de cloro residual adecuado en agua")
st.markdown("#### Ingrese los valores de las variables para predecir si un establecimiento tiene cloro residual adecuado (1 = Sí, 0 = No).")

st.sidebar.header("Campos a Evaluar")

# Inputs para variables categóricas binarias con key
Flag_Menos2anios_Eleccion = st.sidebar.selectbox("Flag Menos 2 años Elección", [0,1], key="Flag_Menos2anios_Eleccion")
Establecimiento_Vigila_Calidad_Agua_SI = st.sidebar.selectbox("Establecimiento Vigila Calidad Agua", [0,1], key="Establecimiento_Vigila_Calidad_Agua_SI")
ApoyoMuni_Control_Calidad_Agua_SI = st.sidebar.selectbox("Apoyo Muni Control Calidad Agua", [0,1], key="ApoyoMuni_Control_Calidad_Agua_SI")
Flag_Miembros_SI_SecundariaSuperior = st.sidebar.selectbox("Flag Miembros Secundaria/Superior", [0,1], key="Flag_Miembros_SI_SecundariaSuperior")
Registro_Cloro_Residual_SI = st.sidebar.selectbox("Registro Cloro Residual", [0,1], key="Registro_Cloro_Residual_SI")
Realizan_Limpieza_SitemaAgua_SI = st.sidebar.selectbox("Realizan Limpieza Sistema Agua", [0,1], key="Realizan_Limpieza_SitemaAgua_SI")
Apoyo_Infraestructura = st.sidebar.selectbox("Apoyo Infraestructura", [0,1], key="Apoyo_Infraestructura")
Apoyo_Tecnico = st.sidebar.selectbox("Apoyo Técnico", [0,1], key="Apoyo_Tecnico")

# Input para variable categórica nominal
cuotapromedio_Cat = st.sidebar.selectbox("Cuota Promedio Categoría", cuotapromedio_options, key="cuotapromedio_Cat")

# Input para variable numérica
Salud_Financiera_Cuotas_Porc = st.sidebar.number_input("Salud Financiera Cuotas (%)", min_value=0.0, max_value=100.0, value=0.0, key="Salud_Financiera_Cuotas_Porc")

# Botón para predecir
if st.sidebar.button("Predecir"):
    # Crear DataFrame para el modelo
    entrada = pd.DataFrame({
        "Flag_Menos2anios_Eleccion": [st.session_state.Flag_Menos2anios_Eleccion],
        "Establecimiento_Vigila_Calidad_Agua_SI": [st.session_state.Establecimiento_Vigila_Calidad_Agua_SI],
        "ApoyoMuni_Control_Calidad_Agua_SI": [st.session_state.ApoyoMuni_Control_Calidad_Agua_SI],
        "cuotapromedio_Cat": [st.session_state.cuotapromedio_Cat],
        "Flag_Miembros_SI_SecundariaSuperior": [st.session_state.Flag_Miembros_SI_SecundariaSuperior],
        "Registro_Cloro_Residual_SI": [st.session_state.Registro_Cloro_Residual_SI],
        "Realizan_Limpieza_SitemaAgua_SI": [st.session_state.Realizan_Limpieza_SitemaAgua_SI],
        "Apoyo_Infraestructura": [st.session_state.Apoyo_Infraestructura],
        "Salud_Financiera_Cuotas_Porc": [st.session_state.Salud_Financiera_Cuotas_Porc/100],
        "Apoyo_Tecnico": [st.session_state.Apoyo_Tecnico]
    })

    st.write("DataFrame de Entradas para el modelo:")
    st.write(entrada)

    # Predicción
    pred_binaria = modelo_cloro.predict(entrada)[0]
    pred_prob = modelo_cloro.predict_proba(entrada)[0][1]

    # Mostrar resultados
    st.markdown(f'<p style="font-size: 40px; color: green;">Predicción binaria de Cloro residual adecuado(Si = 1, No = 0): <strong>{pred_binaria}</strong></p>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-size: 40px; color: green;">Probabilidad de que tenga cloro residual adecuado: <strong>{pred_prob*100:.2f}%</strong></p>', unsafe_allow_html=True)

# Función para resetear los inputs
def reset_inputs():
    st.session_state.Flag_Menos2anios_Eleccion = 0
    st.session_state.Establecimiento_Vigila_Calidad_Agua_SI = 0
    st.session_state.ApoyoMuni_Control_Calidad_Agua_SI = 0
    st.session_state.cuotapromedio_Cat = cuotapromedio_options[0]
    st.session_state.Flag_Miembros_SI_SecundariaSuperior = 0
    st.session_state.Registro_Cloro_Residual_SI = 0
    st.session_state.Realizan_Limpieza_SitemaAgua_SI = 0
    st.session_state.Apoyo_Infraestructura = 0
    st.session_state.Salud_Financiera_Cuotas_Porc = 0.0
    st.session_state.Apoyo_Tecnico = 0

# Botón para resetear
st.sidebar.button("Resetear", on_click=reset_inputs)
