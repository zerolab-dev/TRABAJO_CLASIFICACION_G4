import streamlit as st
import pandas as pd
from joblib import load
import numpy as np

# Cargar modelo entrenado de clasificación (XGBoost, por ejemplo)
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

# Valores posibles para inputs categóricos
cuotapromedio_options = ["_1", "_2", "_3", "_4", "_5"]

st.set_page_config(layout="wide")
st.title("Predicción de Cloro Adecuado en Agua")
st.markdown("Ingrese los valores de las variables para predecir si un establecimiento tiene cloro adecuado (1 = sí, 0 = no).")

st.sidebar.header("Variables del modelo")

# Inputs para variables categóricas binarias
Flag_Menos2anios_Eleccion = st.sidebar.selectbox("Flag Menos 2 años Elección", [0,1])
Establecimiento_Vigila_Calidad_Agua_SI = st.sidebar.selectbox("Establecimiento Vigila Calidad Agua", [0,1])
ApoyoMuni_Control_Calidad_Agua_SI = st.sidebar.selectbox("Apoyo Muni Control Calidad Agua", [0,1])
Flag_Miembros_SI_SecundariaSuperior = st.sidebar.selectbox("Flag Miembros Secundaria/Superior", [0,1])
Registro_Cloro_Residual_SI = st.sidebar.selectbox("Registro Cloro Residual", [0,1])
Realizan_Limpieza_SitemaAgua_SI = st.sidebar.selectbox("Realizan Limpieza Sistema Agua", [0,1])
Apoyo_Infraestructura = st.sidebar.selectbox("Apoyo Infraestructura", [0,1])
Apoyo_Tecnico = st.sidebar.selectbox("Apoyo Técnico", [0,1])

# Inputs para variables categóricas nominales
cuotapromedio_Cat = st.sidebar.selectbox("Cuota Promedio Categoría", cuotapromedio_options)

# Inputs para variables numéricas
Salud_Financiera_Cuotas_Porc = st.sidebar.number_input("Salud Financiera Cuotas (%)", min_value=0.0, max_value=100.0, value=0.0)

# Crear DataFrame para el modelo
entrada = pd.DataFrame({
    "Flag_Menos2anios_Eleccion": [Flag_Menos2anios_Eleccion],
    "Establecimiento_Vigila_Calidad_Agua_SI": [Establecimiento_Vigila_Calidad_Agua_SI],
    "ApoyoMuni_Control_Calidad_Agua_SI": [ApoyoMuni_Control_Calidad_Agua_SI],
    "cuotapromedio_Cat": [cuotapromedio_Cat],
    "Flag_Miembros_SI_SecundariaSuperior": [Flag_Miembros_SI_SecundariaSuperior],
    "Registro_Cloro_Residual_SI": [Registro_Cloro_Residual_SI],
    "Realizan_Limpieza_SitemaAgua_SI": [Realizan_Limpieza_SitemaAgua_SI],
    "Apoyo_Infraestructura": [Apoyo_Infraestructura],
    "Salud_Financiera_Cuotas_Porc": [Salud_Financiera_Cuotas_Porc],
    "Apoyo_Tecnico": [Apoyo_Tecnico]
})

# Botón para predecir
if st.sidebar.button("Predecir"):
    # Predicción
    pred_binaria = modelo_cloro.predict(entrada)[0]
    pred_prob = modelo_cloro.predict_proba(entrada)[0][1]

    # Mostrar resultados
    st.markdown(f"### Resultado de la Predicción")
    st.write(f"Predicción binaria (Cloro adecuado = 1, No = 0): **{pred_binaria}**")
    st.write(f"Probabilidad de que tenga cloro adecuado: **{pred_prob*100:.2f}%**")

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

# Botón para resetear al final del sidebar
st.sidebar.button("Resetear Campos", on_click=reset_inputs)