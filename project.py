
# IRIS SPECIES CLASSIFICATION DASHBOARD

# Students: Juan Sebastian Caro Molina, Edy Alberto Corro Noguera, Juan Bautista Perez Fragoso y Jhoss
# Course: Data Mining - Universidad de la Costa
# Professor: José Escorcia-Gutiérrez, Ph.D.

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# CONFIGURACIÓN STREAMLIT
st.set_page_config(
    page_title="Iris Classifier",
    page_icon="🌸",
    layout="wide"
)

st.title("IRIS SPECIES CLASSIFICATION")
st.write("Dashboard interactivo para clasificar especies de flores Iris usando Machine Learning.")

# CARGA DEL DATASET (TU CSV)
df = pd.read_csv("Iris.csv")

# Renombrar columnas para que coincidan con formatos estándar
df = df.rename(columns={
    "SepalLengthCm": "sepal_length",
    "SepalWidthCm": "sepal_width",
    "PetalLengthCm": "petal_length",
    "PetalWidthCm": "petal_width",
    "Species": "species"
})

# Eliminar columna Id si está presente
df = df.drop(columns=["Id"], errors="ignore")

st.subheader("Dataset Original")
st.dataframe(df)

# PREPROCESAMIENTO
X = df.drop("species", axis=1)
y = df["species"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# División train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# MODELO: RANDOM FOREST
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# MÉTRICAS
st.subheader("📊 Model Performance Metrics")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
c2.metric("Precision", f"{precision_score(y_test, y_pred, average='macro'):.3f}")
c3.metric("Recall", f"{recall_score(y_test, y_pred, average='macro'):.3f}")
c4.metric("F1 Score", f"{f1_score(y_test, y_pred, average='macro'):.3f}")

st.divider()

# VISUALIZACIONES EXPLICATIVAS
st.subheader("Histogramas de Características")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
sns.histplot(df["sepal_length"], kde=True, ax=axes[0][0])
sns.histplot(df["sepal_width"], kde=True, ax=axes[0][1])
sns.histplot(df["petal_length"], kde=True, ax=axes[1][0])
sns.histplot(df["petal_width"], kde=True, ax=axes[1][1])
st.pyplot(fig)

st.divider()

# PREDICCIÓN DE USUARIO
st.subheader("Predicción de Nueva Flor")

sl = st.slider("Sepal Length", 4.0, 8.0, 5.5)
sw = st.slider("Sepal Width", 2.0, 5.0, 3.0)
pl = st.slider("Petal Length", 1.0, 7.0, 4.0)
pw = st.slider("Petal Width", 0.1, 3.0, 1.2)

new_data = pd.DataFrame([[sl, sw, pl, pw]], columns=X.columns)
scaled_new = scaler.transform(new_data)

prediction = model.predict(scaled_new)[0]

st.success(f"🌸 **La especie predicha es: {prediction.upper()}**")

# VISUALIZACIÓN 3D DEL NUEVO PUNTO
st.subheader("🌐 Visualización 3D del Nuevo Ejemplo")

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    df["sepal_length"], df["sepal_width"], df["petal_length"],
    c=df["species"].astype('category').cat.codes, cmap="viridis", s=50
)

ax.scatter(sl, sw, pl, c="red", s=200)

ax.set_xlabel("Sepal Length")
ax.set_ylabel("Sepal Width")
ax.set_zlabel("Petal Length")
st.pyplot(fig)
