import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Asumiendo que tu DataFrame se llama 'df' y ya está cargado
# df = pd.read_csv('telco_customer_churn.csv')
## url = 'https://raw.githubusercontent.com/Sarthakrshetty/Telco-Customer-Churn-Analysis/refs/heads/main/customer%20churn.csv'
df = pd.read_csv(r'C:\Users\REINALDO\Videos\Proyecto\customer_churn.csv')

# --- 1. Análisis de la Variable Objetivo (Churn) ---

print("--- 1. Análisis del Desbalance (Variable 'Churn') ---")

# Calculamos el número de clientes que abandonaron y los que no
churn_counts = df['Churn'].value_counts()
churn_percent = df['Churn'].value_counts(normalize=True) * 100

print("\nConteo de Clientes:")
print(churn_counts)
print("\nPorcentaje de Churn:")
print(round(churn_percent, 2), '%')

# Visualización del Desbalance
plt.figure(figsize=(6, 4))
sns.barplot(x=churn_counts.index, y=churn_counts.values)
plt.title('Distribución de Clientes con y sin Churn')
plt.xlabel('Churn')
plt.ylabel('Conteo de Clientes')
plt.show()

print("Observación Clave: El dataset está desbalanceado. La tasa de abandono es del", 
      round(churn_percent['Yes'], 2), "%")

# --- 2. Análisis de la Variable Numérica Clave: Antigüedad (Tenure) ---

print("\n--- 2. Análisis de la Antigüedad (Tenure) vs. Churn ---")

# Gráfico de Caja (Box Plot) para comparar la distribución de Tenure
plt.figure(figsize=(8, 6))
sns.boxplot(x='Churn', y='tenure', data=df)
plt.title('Distribución de Antigüedad (Tenure) por Clase de Churn')
plt.xlabel('Churn (Abandono)')
plt.ylabel('Antigüedad (Meses)')
plt.show()

# Gráfico de Histograma para ver la distribución completa
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='tenure', hue='Churn', multiple='stack', bins=30, kde=True)
plt.title('Histograma de Antigüedad por Churn')
plt.xlabel('Antigüedad (Meses)')
plt.ylabel('Frecuencia')
plt.show()

# Calcular la Antigüedad Promedio (para el README)
tenure_avg = df.groupby('Churn')['tenure'].mean().round(2)
print("\nAntigüedad (Tenure) Promedio:")
print(tenure_avg)

print("\nObservación Clave: Los clientes que abandonan ('Yes') tienen una antigüedad promedio de:", 
      tenure_avg['Yes'], "meses, significativamente menor a los que se quedan.")

# Continuando con el DataFrame 'df'

print("--- 3. Análisis de Variables Categóricas Clave vs. Churn ---")

# Lista de variables categóricas a analizar
categorias_clave = ['Contract', 'OnlineSecurity', 'PaymentMethod']

# Función para calcular y visualizar la tasa de Churn por categoría
def analizar_categoria(df, columna):
    # Calcular el porcentaje de Churn ('Yes') para cada categoría
    churn_rate = df.groupby(columna)['Churn'].value_counts(normalize=True).mul(100).unstack()['Yes'].sort_values(ascending=False)
    
    print(f"\nTasas de Churn por {columna}:")
    print(round(churn_rate, 2), '%')

    # Visualización (Gráfico de barras apiladas)
    plt.figure(figsize=(8, 5))
    df_plot = df.groupby(columna)['Churn'].value_counts(normalize=True).mul(100).unstack()
    df_plot.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.title(f'Distribución de Churn por {columna}')
    plt.ylabel('Porcentaje')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Churn')
    plt.show()

# Ejecutar el análisis para cada categoría
for col in categorias_clave:
    analizar_categoria(df, col)
    

import numpy as np

# Convertir 'TotalCharges' a numérico. El argumento 'coerce' convierte los errores (como strings vacíos) en NaN.
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Verificamos cuántos NaN hay (deberían ser muy pocos)
print("Valores NaN en TotalCharges:", df['TotalCharges'].isnull().sum())

# La mejor forma de imputar es con la mediana, pero dado que son clientes muy nuevos (tenure=0),
# lo más seguro es imputarlos con 0, ya que no han generado cargos totales.

# Si decides imputar con 0 (basado en el contexto de churn dataset)
df['TotalCharges'].fillna(0, inplace=True)

print("Verificación de valores faltantes después de la imputación:", df['TotalCharges'].isnull().sum())





# Variables binarias que necesitan ser convertidas a 0 y 1
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity',
               'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
               'StreamingMovies', 'PaperlessBilling', 'Churn']

for col in binary_cols:
    # Mapeo simple de 'Yes' a 1 y 'No' a 0, asumiendo que el 'No internet service'
    # y 'No phone service' se tratarán en el One-Hot Encoding si existen o se convierten a 'No'.
    # Para el Target 'Churn', 'Yes' es la clase positiva (1).
    if col == 'Churn':
        df[col] = df[col].map({'Yes': 1, 'No': 0})
    else:
        df[col] = df[col].replace({'Yes': 1, 'No': 0, 'No internet service': 0, 'No phone service': 0})


# Variables multi-categóricas
categorical_cols = ['gender', 'InternetService', 'Contract', 'PaymentMethod']

# Aplicar One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True) 
# drop_first=True elimina una columna de cada grupo para evitar multicolinealidad.
# Por ejemplo, si tienes 'Gender_Female' y 'Gender_Male', solo necesitas una columna.

# Eliminamos la columna original 'customerID' ya que no es útil para el modelo
df_final = df_encoded.drop(columns=['customerID'])

print("\nDimensiones del DataFrame después de la codificación:", df_final.shape)
print("Primeras 5 filas del DataFrame final (muestra de codificación):")
print(df_final.head())


# 1. Separar características (X) y objetivo (y)
X = df_final.drop('Churn', axis=1) # Todas las columnas excepto 'Churn'
y = df_final['Churn']              # Solo la columna 'Churn'



from sklearn.model_selection import train_test_split

# Dividir los datos en conjuntos de entrenamiento (80%) y prueba (20%)
# Usamos stratify=y para asegurar que ambos conjuntos tengan la misma proporción de Churn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\nDimensiones de los conjuntos:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")


from sklearn.preprocessing import StandardScaler

# Identificar las columnas a escalar (las numéricas originales)
# 'SeniorCitizen' es binaria (0/1), por lo que no necesita escalado.
cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Inicializar el escalador
scaler = StandardScaler()

# Ajustar (fit) el escalador solo en el conjunto de ENTRENAMIENTO y transformarlo
X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])

# Solo transformar (transform) el conjunto de PRUEBA (usando la media y desviación estándar del entrenamiento)
X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

print("\nVerificación de datos escalados (X_train):")
print(X_train[cols_to_scale].head())


import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

print("--- 1. Aplicando SMOTE y Entrenando XGBoost ---")

# Debido al desbalance de clases (visto en el EDA), aplicamos SMOTE solo al conjunto de entrenamiento.
# Esto genera datos sintéticos de la clase minoritaria ('Churn=Yes').
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"\nProporción Churn en datos originales: {y_train.sum() / len(y_train) * 100:.2f}%")
print(f"Proporción Churn en datos con SMOTE: {y_train_smote.sum() / len(y_train_smote) * 100:.2f}%")

# Inicializar y entrenar el clasificador XGBoost
# Nota: La estructura de XGBoost compensa un poco el desbalance, pero el SMOTE ayuda mucho en este caso.
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic', # Clasificación binaria
    eval_metric='logloss',       # Métrica para evaluación interna
    use_label_encoder=False,     # Práctica recomendada por XGBoost
    random_state=42
)

# Entrenar el modelo con los datos balanceados
xgb_model.fit(X_train_smote, y_train_smote)
print("\nModelo XGBoost entrenado con éxito.")



# Realizar predicciones en el conjunto de prueba
y_pred = xgb_model.predict(X_test)
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1] # Probabilidades para ROC AUC

print("\n--- 2. Evaluación del Modelo en Conjunto de Prueba ---")

# Informe de Clasificación
print("\nInforme de Clasificación:")
print(classification_report(y_test, y_pred))

# Matriz de Confusión
cm = confusion_matrix(y_test, y_pred)
print("Matriz de Confusión:")
print(cm)

# ROC AUC Score (Mide qué tan bien el modelo distingue entre las clases)
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nROC AUC Score: {roc_auc:.4f}")

# Explicación de las métricas
print("\nAnálisis de Métricas:")
print("🎯 El 'Recall' para la clase 1 (Churn=Yes) es la métrica más crítica en negocio, ya que mide cuántos clientes en riesgo REAL el modelo logró capturar.")
print("⭐ El F1-Score ofrece una visión balanceada de la Precisión y el Recall.")




import shap

print("\n--- 3. Explicabilidad con SHAP ---")

# 1. Crear el 'explainer' de SHAP (usa TreeExplainer para modelos basados en árboles)
explainer = shap.TreeExplainer(xgb_model)

# 2. Calcular los valores SHAP para el conjunto de prueba
shap_values = explainer.shap_values(X_test)

# 3. Visualizar la importancia global de las características (Summary Plot)
print("\nGráfico de Importancia Global (Summary Plot):")
# Cada punto es una predicción de un cliente. 
# El color indica el valor de la característica (rojo = alto, azul = bajo).
# El eje X es el impacto en la predicción.
shap.summary_plot(shap_values, X_test)
# 

print("Interpretación Global: El gráfico anterior muestra que los factores como el 'Tipo de Contrato (mes a mes)', la 'Antigüedad (tenure)' y el 'Servicio de Internet (Fibra óptica)' son los principales impulsores del Churn.")



# --- 4. Selección del Cliente de Alto Riesgo ---

# 1. Obtenemos las probabilidades predichas en el conjunto de prueba
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

# 2. Encontramos la POSICIÓN (no el índice) del primer cliente con probabilidad > 0.9
# Usamos np.where para encontrar la posición en el array
high_risk_position_in_test = np.where(y_pred_proba > 0.9)[0].tolist()[0] 
# Si el modelo no tiene predicciones > 0.9, necesitarás ajustar este umbral.

# 3. Accedemos a los datos del cliente de PRUEBA usando .iloc (posición)
customer_data = X_test.iloc[high_risk_position_in_test]
# customer_data será la fila de entrada para la explicación

print(f"\nExplicación Local para Cliente de Alto Riesgo (Posición en X_test: {high_risk_position_in_test}):")
print("Características del cliente (Primeras 10):")
print(customer_data.head(10)) 

# --- 5. Visualización del Gráfico de Cascada Corregida ---

# 1. Creamos el objeto EXPLANATION completo para el conjunto de prueba.
# Esto solo se hace una vez, pero lo repetimos aquí para asegurarnos de que el 'explainer' esté en el contexto.
explainer_output = explainer(X_test)

# 2. Usamos el objeto de EXPLICACIÓN y accedemos a la posición específica.
# Al pasar el objeto completo explainer_output[posición], se pasa la base value, los SHAP values,
# y los valores de las características, tal como lo requiere la función waterfall.
shap.plots.waterfall(
    explainer_output[high_risk_position_in_test],
    max_display=10,
    show=True
)

print("\nInterpretación Local: El gráfico de cascada muestra cómo cada característica específica de este cliente (en rojo) impulsó la predicción hacia la clase 'Churn=Yes'.")


import joblib
# Después de entrenar:
joblib.dump(xgb_model, 'xgb_model.pkl')
joblib.dump(scaler, 'scaler.pkl')


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import io
import streamlit.components.v1 as components


# ----------------------------------------------------
# 0. FUNCIÓN SHAP UNIVERSAL (HTML + MATPLOTLIB)
# ----------------------------------------------------
def st_shap(plot, height=250):
    """
    Renderiza cualquier gráfico SHAP en Streamlit.
    - Si es HTML (ej. force_plot) → lo incrusta con components.html
    - Si es Matplotlib (ej. waterfall) → lo convierte a imagen PNG
    """
    # Caso 1: Gráfico HTML (force_plot)
    if hasattr(plot, "html"):
        shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
        components.html(shap_html, height=height)
        return

    # Caso 2: Gráfico Matplotlib (waterfall, beeswarm, bar, etc.)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    st.image(buf)
    plt.clf()



# ----------------------------------------------------
# 1. CONFIGURACIÓN E IMPORTACIÓN DE MODELOS
# ----------------------------------------------------
st.set_page_config(page_title="Predicción de Churn", layout="wide")

try:
    model = joblib.load('xgb_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_names = model.get_booster().feature_names
except FileNotFoundError:
    st.error("❌ Error: Archivos xgb_model.pkl o scaler.pkl no encontrados.")
    st.stop()



# ----------------------------------------------------
# 2. INTERFAZ DE USUARIO
# ----------------------------------------------------
st.title("👨‍💻 Predictor de Abandono de Clientes (Churn)")
st.subheader("Herramienta basada en XGBoost y análisis SHAP.")

with st.sidebar:
    st.header("Características del Cliente")

    tenure = st.slider("Antigüedad (Meses)", 0, 72, 24)
    monthly_charges = st.number_input("Cargos Mensuales ($", 18.25, 118.75, 50.0)

    contract = st.selectbox("Tipo de Contrato", ['Month-to-month', 'One year', 'Two year'])
    internet_service = st.selectbox("Servicio de Internet", ['Fiber optic', 'DSL', 'No'])
    security = st.selectbox("Seguridad en Línea", ['Yes', 'No', 'No internet service'])

predict_button = st.button("🚀 Predecir y Explicar el Churn")



# ----------------------------------------------------
# 3. LÓGICA DE PREDICCIÓN
# ----------------------------------------------------
if predict_button:

    # --- 3.1 Crear vector vacío con columnas del modelo ---
    input_data = pd.Series(0, index=feature_names)

    # --- 3.2 Asignar valores numéricos ---
    input_data['tenure'] = tenure
    input_data['MonthlyCharges'] = monthly_charges

    if 'TotalCharges' in input_data:
        input_data['TotalCharges'] = tenure * monthly_charges

    # --- 3.3 One-Hot Encoding manual según las columnas existentes ---
    if contract == 'One year':
        input_data['Contract_One year'] = 1
    elif contract == 'Two year':
        input_data['Contract_Two year'] = 1

    if internet_service == 'Fiber optic':
        input_data['InternetService_Fiber optic'] = 1
    elif internet_service == 'No':
        input_data['InternetService_No'] = 1

    # Seguridad en línea
    if security == 'Yes':
        if 'OnlineSecurity_Yes' in input_data:
            input_data['OnlineSecurity_Yes'] = 1
        elif 'OnlineSecurity' in input_data:
            input_data['OnlineSecurity'] = 1
    elif security == 'No internet service':
        if 'OnlineSecurity_No internet service' in input_data:
            input_data['OnlineSecurity_No internet service'] = 1

    # --- 3.4 Crear DataFrame final ---
    X_input = pd.DataFrame([input_data.values], columns=feature_names)

    # --- 3.5 Escalar variables numéricas ---
    cols_to_scale = [col for col in ['tenure', 'MonthlyCharges', 'TotalCharges'] if col in X_input.columns]
    if cols_to_scale:
        X_input[cols_to_scale] = scaler.transform(X_input[cols_to_scale])

    # --- 3.6 Predicción ---
    prediction_proba = model.predict_proba(X_input)[:, 1][0]

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        if prediction_proba >= 0.5:
            st.error("⚠️ **ALTO RIESGO DE ABANDONO (CHURN)**")
        else:
            st.success("✅ **BAJO RIESGO DE ABANDONO (CHURN)**")

    with col2:
        st.metric("Probabilidad estimada", f"{prediction_proba*100:.2f}%")



    # ----------------------------------------------------
    # 4. EXPLICACIÓN SHAP
    # ----------------------------------------------------
    st.subheader("🧠 Explicación de la Predicción (SHAP)")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_input)

    plt.figure(figsize=(6, 4)) # Ajustar el tamaño del gráfico
    shap_plot = shap.plots.waterfall(
        shap_values[0],
        max_display=10,
        show=False
    )

    st_shap(shap_plot, height=250)

    st.info("El gráfico explica cómo cada característica aumentó (rojo) o disminuyó (azul) el riesgo de abandono.")


