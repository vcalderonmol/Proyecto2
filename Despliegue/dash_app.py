#Sistemax
import os
import io
import base64
from dotenv import load_dotenv

#dash
import dash
import seaborn as sns
from dash import dcc # dash core components
from dash import html # dash html components
from dash.dependencies import Input, Output, State
from typing import List, Tuple

#Manejo de datos
import psycopg2  # type: ignore
import pandas as pd
import numpy as np
import dill
import plotly.express as px

#conectar base de datos
engine = psycopg2.connect(
    dbname='credit_default',
    user='postgres',
    password='postgres2024',
    host='proyectos.c1oweiqq27pm.us-east-1.rds.amazonaws.com',
    port='5432'
)
query = """
    SELECT limit_bal, sex, marriage, age, education, default_payment_next_month
    FROM customer_credit
    WHERE marriage <> '0'
"""
datos_description = pd.read_sql(query, engine)
#Correlación con variable Y
correlaciones = datos_description.corr().iloc[0:5,-1]
#datos_description = datos_description.sort_values(by=datos_description.columns[0], ascending=False)
correlaciones = correlaciones.T
# Crear una lista para almacenar los divs generados dinámicamente
divs = []
# Iterar sobre los valores de la serie datos_description
for columna, correlacion in correlaciones.items():
    # Calcular la opacidad del color de fondo
    opacity = abs(correlacion)*3

    # Definir el estilo del div con la opacidad ajustada
    estilo = {
        'font-weight': 'bold',
        'font-size':' 16px',
        'text-align': 'center',
        'margin': '2px',
        'box-shadow': '2px 2px 5px grey',
        'border-radius': '5px',
        'padding': '5px',
        'width': '40%',
        'background': f'rgb(99,110,247, {opacity})'  # Color de fondo morado con opacidad ajustada
    }

    # Crear el contenido del div
    contenido = f"{columna}"
    contenido2 = f"{int(correlacion* 100)}%"

    # Agregar el div a la lista divs
    div = html.Div([contenido, html.Br(),contenido2 ], style=estilo)
    divs.append(div)

#Gráficos descriptivos---
datos_filtrados = datos_description[['limit_bal', 'sex', 'marriage', 'default_payment_next_month']]

# Crear el gráfico de dispersión entre limit_bal y default_payment_next_month con colores personalizados
scatter_fig = px.scatter(datos_filtrados, x='limit_bal', y='default_payment_next_month',
                        title='Default Payments Vs Límite de Crédito',
                        labels={'default_payment_next_month': 'Default Payment Next Month'})

# Establecer el color y opacidad personalizados en el gráfico
scatter_fig.update_traces(marker=dict(color=datos_filtrados['default_payment_next_month'].map({0: 'rgba(99, 110, 247, 1)', 1: 'rgba(99, 110, 247, 0.01)'})))

# Quitar la leyenda de colores
scatter_fig.update_layout(showlegend=False)

# Crear el gráfico de barras para contar los impagos por cada tipo de matrimonio 'Cuenta Default Payments (0) por tipo de Matrimonio'
bar_fig = px.histogram(datos_filtrados[(datos_filtrados['default_payment_next_month'] == 0)
                                & (datos_filtrados['marriage'] != '0')],
                x='marriage',
                nbins=4,
                title='Cantidad de default payments por tipo de matrimonio')

graficos = html.Div(
            [html.Div(dcc.Graph(figure=scatter_fig), style={'flex': '1'}),
            html.Div(dcc.Graph(figure=bar_fig), style={'flex': '1'})],
            style={'display': 'flex', 'flex-direction': 'row'})

#Métodos para cargar el modelo y realizar predicciones
def cargar_modelo(ruta_archivo):
    # Cargar el modelo desde el archivo con dill
    with open(ruta_archivo, 'rb') as f:
        modelo = dill.load(f)
    return modelo

def predecir(datos):
    predicciones_proba = []
    predicciones_binarias = []
    # Obtener las probabilidades de predicción de cada modelo en los datos de entrada
    for modelo in modelos:
        try:
            predicciones_binarias.append(modelo.predict(datos))
            predicciones_proba.append(modelo.predict_proba(datos))
        except Exception as e:
            # Imprimir el mensaje de error en la consola
            print("Error en el modelo:", e)
    # Calcular el promedio de las probabilidades de predicción de todos los modelos
    average_proba_predictions = np.mean(predicciones_proba, axis=0)
    # Convertir las probabilidades promedio en predicciones binarias utilizando un umbral de decisión de 0.5
    predicciones_binarias = np.mean(predicciones_binarias, axis=0)
    predicciones_binarias = np.where(predicciones_binarias >= 0.5, 1, 0)
    return predicciones_binarias, average_proba_predictions


#Definir la aplicación de Dash
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# Cargar el modelo desde el archivo
modelo_apilado = cargar_modelo("Modelos/pipeline_apilado.pkl")
modelo_goteo = cargar_modelo("Modelos/pipeline_goteo.pkl")
modelo_piramide = cargar_modelo("Modelos/pipeline_piramide.pkl")
modelo_piramide_inv = cargar_modelo("Modelos/pipeline_piramide_inv.pkl")
modelo_conv = cargar_modelo("Modelos/pipeline_conv.pkl")
modelos = [modelo_apilado, modelo_goteo, modelo_piramide, modelo_piramide_inv]

# Crear el layout de la aplicación
app.layout = html.Div(
    [
        html.H6("Descripción Clientes: Default Credit Card Payments", style={'font-weight': 'bold'}),
        html.Br(),
        html.Div(html.H6("Correlación con Default Payments"), style={'margin-left': '50px','font-weight': 'bold','font-size':' 18px',}),
        html.Div(divs, style = {'width': '60%', 'display': 'flex', 'flex-direction': 'row', 'align-items': 'center','margin-left': '50px'}),
        html.Div(graficos),
        html.Br(),
        html.H6("Reporte Predicción: Default Credit Card Payments", style={'font-weight': 'bold'}),
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
    predicciones, predicciones_proba = predecir(datos)
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
            ],style={'margin': '20px','box-shadow': '2px 2px 5px grey', 'border-radius': '5px', 'padding': '5px','width': '60%'}),
            html.Div([
                html.Div([
                    html.Div([
                        html.H6("# Clientes Propensos a Impago", style={'display': 'inline-block', 'font-weight': 'bold', 'width': '80%','text-align': 'center'}),
                        html.H6(f"Nominal: {inpagos}", style={'display': 'inline-block', 'margin-left': '10px', 'color': 'rgb(99,110,247, 1)'}),
                        html.H6(f"Porcentual: {inpagos_porcentaje}%", style={'display': 'inline-block', 'margin-left': '10px', 'color': 'rgb(99,110,247, 1)'})
                    ], style={'display': 'flex', 'flex-direction': 'column','align-items': 'center', 'margin': '20px','box-shadow': '2px 2px 5px grey', 'border-radius': '5px', 'padding': '5px', 'width': '60%'}),
                    html.Div([
                        html.H6("# Clientes Propensos a Pago", style={'display': 'inline-block', 'font-weight': 'bold', 'width': '80%','text-align': 'center'}),
                        html.H6(f"Nominal: {pagos}", style={'display': 'inline-block', 'margin-left': '10px', 'color': 'rgb(99,110,247, 0.5)'}),
                        html.H6(f"Porcentual: {pagos_porcentaje}%", style={'display': 'inline-block', 'margin-left': '10px', 'color': 'rgb(99,110,247, 0.5)'})
                    ], style={'display': 'flex', 'flex-direction': 'column','align-items': 'center', 'margin': '20px','box-shadow': '2px 2px 5px grey', 'border-radius': '5px', 'padding': '5px','width': '60%'})
                ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center'}),
                html.Div([
                    html.Div([
                        html.H6("Probabilidad de Impago Promedio", style={'display': 'inline-block', 'font-weight': 'bold', 'width': '80%','text-align': 'center'}),
                        html.H6(f"{proba_impago}%", style={'display': 'inline-block', 'margin-left': '10px', 'color': 'rgb(99,110,247, 1)'})
                    ], style={'display': 'flex', 'flex-direction': 'column','align-items': 'center', 'margin': '20px','box-shadow': '2px 2px 5px grey', 'border-radius': '5px', 'padding': '5px', 'width': '60%'}),
                    html.Div([
                        html.H6("Probabilidad de Pago Promedio", style={'display': 'inline-block', 'font-weight': 'bold', 'width': '80%','text-align': 'center'}),
                        html.H6(f"{proba_pago}%", style={'display': 'inline-block', 'margin-left': '10px', 'color': 'rgb(99,110,247, 0.5)'})
                    ], style={'display': 'flex', 'flex-direction': 'column','align-items': 'center', 'margin': '20px','box-shadow': '2px 2px 5px grey', 'border-radius': '5px', 'padding': '5px', 'width': '60%'})
                ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center'})
            ])
        ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center'} )
    ])


    return [content_div]

#ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(host = "0.0.0.0", debug=True, port=8050)

