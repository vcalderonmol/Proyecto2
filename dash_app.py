#Sistemax
import os
import io
import base64
from dotenv import load_dotenv

#dash
import dash
from dash import dcc # dash core components
from dash import html # dash html components
from dash.dependencies import Input, Output, State
from typing import List, Tuple

#Manejo de datos
#import psycopg2  # type: ignore
import pandas as pd
import numpy as np
import dill
import plotly.express as px


def cargar_modelo(ruta_archivo):
    # Cargar el modelo desde el archivo con dill
    with open(ruta_archivo, 'rb') as f:
        modelo = dill.load(f)
    return modelo

def predecir(modelo, datos):
    # Realizar las predicciones
    predicciones_proba = modelo.predict_proba(datos)
    predicciones = modelo.predict(datos)
    return predicciones, predicciones_proba

#Definir la aplicación de Dash
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# Cargar el modelo desde el archivo
modelo_apilado = cargar_modelo("Modelos/pipeline_apilado.pkl")

# Crear el layout de la aplicación
app.layout = html.Div(
    [
        html.H6("Default Credit Card Clients", style={'font-weight': 'bold'}),
        html.Br(),
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Arrastre y suelte o ',
                html.A('Seleccione un archivo CSV')
            ]),
            style={
                'width': '50%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '5px'
            },
            # Permitir múltiples archivos
            multiple=False
        ),
        html.Br(),
        html.Div(id='output-data-upload')
    ]
)

# Callback para cargar los datos del archivo CSV
@app.callback([Output('output-data-upload', 'children')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def update_output(contents, filename):
    if contents is None:
        # Si no hay contenido, no hacemos nada y retornamos una lista vacía
        return [html.Div()]

    # Decodificar el contenido del archivo CSV
    content_type, content_string = contents.split(',')
    datos = pd.read_csv(io.StringIO(base64.b64decode(content_string).decode('utf-8')))
    datos = datos.to_numpy()
    predicciones, predicciones_proba = predecir(modelo_apilado, datos)
    #Probabilidad de impago y pago
    promedio_proba = np.mean(predicciones_proba, axis=0)
    proba_impago = round(promedio_proba[0]*100, 2)
    proba_pago = round(promedio_proba[1]*100, 2)
    #Cantidad de pagos e impagos
    pagos = np.count_nonzero(predicciones == 1)
    pagos_porcentaje = round(pagos*100/(len(predicciones)),0)
    inpagos = np.count_nonzero(predicciones == 0)
    inpagos_porcentaje = round(inpagos*100/(len(predicciones)),0)
    # Calcular el histograma de la primera columna de predicciones_proba
    histogram_data = px.histogram(x=predicciones_proba[:, 0], title='Histograma de Probabilidades de Impago', labels={'x': 'Probabilidad de Impago', 'y': 'Frecuencia'})


    # Crear el contenido del Div con la cantidad de pagos e impagos
    content_div = html.Div([
        html.H5(f'Análisis Predictivo Sobre: {filename}', style={'font-weight': 'bold'}),
        html.Div([
            html.Div([
                dcc.Graph(figure=histogram_data)
            ]),
            html.Div([
                html.Div([
                    html.Div([
                        html.H6("# Clientes Propensos a Impago", style={'display': 'inline-block', 'font-weight': 'bold', 'width': '80%','text-align': 'center'}),
                        html.H6(f"{inpagos} ({inpagos_porcentaje}%)", style={'display': 'inline-block', 'margin-left': '10px', 'color': 'red'})
                    ], style={'display': 'flex', 'flex-direction': 'column','align-items': 'center', 'margin': '20px','box-shadow': '2px 2px 5px grey', 'border-radius': '5px', 'padding': '5px', 'width': '60%'}),
                    html.Div([
                        html.H6("# Clientes Propensos a Pago", style={'display': 'inline-block', 'font-weight': 'bold', 'width': '80%','text-align': 'center'}),
                        html.H6(f"{pagos} ({pagos_porcentaje}%)", style={'display': 'inline-block', 'margin-left': '10px', 'color': 'green'})
                    ], style={'display': 'flex', 'flex-direction': 'column','align-items': 'center', 'margin': '20px','box-shadow': '2px 2px 5px grey', 'border-radius': '5px', 'padding': '5px','width': '60%'})
                ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center'}),
                html.Div([
                    html.Div([
                        html.H6("Probabilidad de Impago Promedio", style={'display': 'inline-block', 'font-weight': 'bold', 'width': '80%','text-align': 'center'}),
                        html.H6(f"{proba_impago}%", style={'display': 'inline-block', 'margin-left': '10px', 'color': 'red'})
                    ], style={'display': 'flex', 'flex-direction': 'column','align-items': 'center', 'margin': '20px','box-shadow': '2px 2px 5px grey', 'border-radius': '5px', 'padding': '5px', 'width': '60%'}),
                    html.Div([
                        html.H6("Probabilidad de Pago Promedio", style={'display': 'inline-block', 'font-weight': 'bold', 'width': '80%','text-align': 'center'}),
                        html.H6(f"{proba_pago}%", style={'display': 'inline-block', 'margin-left': '10px', 'color': 'green'})
                    ], style={'display': 'flex', 'flex-direction': 'column','align-items': 'center', 'margin': '20px','box-shadow': '2px 2px 5px grey', 'border-radius': '5px', 'padding': '5px', 'width': '60%'})
                ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center'})
            ])
        ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center'} )
    ])


    return [content_div]

#ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True, port=8050)

