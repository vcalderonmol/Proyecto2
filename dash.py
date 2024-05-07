import pandas as pd
import numpy as np
import dill

def cargar_datos_modelo(ruta_archivo):
    # Cargar el archivo CSV en un DataFrame
    df = pd.read_csv(ruta_archivo)
    # Convertir el DataFrame en un numpy array
    datos_modelo = df.to_numpy()
    # Liberar memoria eliminando el DataFrame
    del df
    return datos_modelo

def cargar_modelo(ruta_archivo):
    # Cargar el modelo desde el archivo con dill
    with open(ruta_archivo, 'rb') as f:
        modelo = dill.load(f)
    return modelo

def predecir(modelo, datos):
    # Realizar las predicciones
    predicciones = modelo.predict_proba(datos)
    return predicciones

# Saludo al usuario
print("¡Hola! Bienvenido.")

# Preguntar la edad al usuario
edad = input("¿Cuál es tu edad? ")

# Cargar los datos del archivo CSV y convertirlos en un numpy array
datos_modelo = cargar_datos_modelo("datos_modelo/test.csv")

# Cargar el modelo desde el archivo
modelo = cargar_modelo("Modelos/pipeline_apilado.pkl")

# Realizar predicciones con el modelo
predicciones = predecir(modelo, datos_modelo)

# Mostrar las primeras 5 predicciones
print("Las primeras 5 predicciones son:", predicciones[:5])
