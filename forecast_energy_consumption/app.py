#from tkinter import font
from jupyter_server import DEFAULT_STATIC_FILES_PATH
from pandas import timedelta_range
import streamlit as st
import datetime
import matplotlib.pyplot as plt
from matplotlib import image
from scipy import misc
from PIL import Image
from datetime import timedelta
from forecast_energy_consumption.dataprep import X_y_train_test
from forecast_energy_consumption.predict import predict_output
import pandas as pd
import numpy as np
from forecast_energy_consumption.consumption_history import consumption_history
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from forecast_energy_consumption.knn_production import knn_production
from forecast_energy_consumption.main import main
import plotly.graph_objects as go

#Main title
st.markdown(f"<h1 style='text-align: center; color: black; font: Roboto'> Prédiction de consommation d'énergie en région PACA</h1>", unsafe_allow_html=True)

st.write('')
st.write('')
st.write('')
st.write('')
st.write('')

url = 'http://127.0.0.1:5000/predict' #'https://taxifare.lewagon.ai/predict' (exercice 1, jeudi)

#Enter the input date
date1 = st.date_input(label='Date de première prévision :',value= datetime.date(2021, 1, 1), min_value=datetime.date(2020, 1, 1), max_value=datetime.date(2022, 4, 30))

st.write('')
st.write('')
st.write('')
st.write('')
st.write('')

date2 = date1 + timedelta(days = 13)

date_test = pd.DataFrame([date1,date2]).set_index(0).asfreq('D') 
df_train, X_test, y_test, predictions, mape, predictions_plus_x_celsius, predictions_moins_x_celsius = main('xgb',str(date1), 14)#(model_name,date_debut_test, nombre_jours_test)

date_train, prod_history = consumption_history(str(date1))

#Energy consumption 1 year before
fig1 = plt.figure(figsize=(10, 4))
fig1 = px.line(date_train, x="Date", y="Consommation (MW)")#,title_x = 5)#,title_size = 10)
fig1.update_layout(title_text=f"Consommation d'énergie<br>entre le {date1 - timedelta(days = 365)} et le {date1}", title_x=0.5,title_y=0.95, font=dict(family="Roboto",size=14,color="black"))#,fontsize = 8)

st.plotly_chart(fig1)

RPA = prod_history

st.write('')
st.write('')
st.write('')
st.write('')
st.write('')

#Graph pie to describe energy production repartition
fig2 = plt.figure(figsize=(10, 4))
fig2 = px.pie(values = np.array(prod_history.values).tolist()[0],names = prod_history.columns)
fig2.update_layout(title_text=f"Répartition de la consommation<br>entre le {date1 - timedelta(days = 365)} et le {date1}", title_x=0.5,title_y=0.96, font=dict(family="Roboto",size=14,color="black"))#,fontsize = 8)
st.plotly_chart(fig2)

st.write('')
st.write('')
st.write('')
st.write('')
st.write('')

list_date = []
date_pour_list = str(date1)

for i in range(0,14):
    list_date.append(date_pour_list)
    Date_datetime = pd.to_datetime(date_pour_list)
    Date_plus_1 =  Date_datetime + timedelta(1)
    date_pour_list = str(Date_plus_1)[0:10]
    
# Prediction graph of energy consumption
fig3 = plt.figure(figsize=(10, 4))

layout = go.Layout(yaxis = dict(title = 'Consommation (MW)',zeroline=True,showline = True),xaxis = dict(title = 'Date', zeroline = True,showline = True),
                   legend=dict(yanchor="top",y=0.99,xanchor="left",x=-10*0.01,font=dict(family="Roboto",size=10,color="black")))

fig3 = go.Figure([go.Scatter(x=date_test.index,y=predictions,line=dict(color='rgb(0,100,80)'),mode='lines',name="Consommation prédite"),
    go.Scatter(x=list_date+list_date[::-1],y=predictions_plus_x_celsius+predictions_moins_x_celsius[::-1], fill='toself',fillcolor='rgba(0,100,80,0.2)',
               line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", showlegend=True, name="marge d'erreur",)],layout=layout)

fig3.add_trace(go.Scatter(x=list_date,y=y_test,name="Consommation réelle", line = go.Line(color = "blue",dash = "dash")))

fig3.update_layout(title_text=f"Prévision de consommation<br>entre le {date1} et le {date2}", title_x=0.5,title_y=0.9, font=dict(family="Roboto",size=14,color="black"))#,fontsize = 8)

st.plotly_chart(fig3)

#Display the error
st.markdown(f"<h4 style='text-align: center; color: green; font: Roboto'>Erreur moyenne = {round(mape*100,2)} %</h4>", unsafe_allow_html=True)

st.write('')
st.write('')
st.write('')
st.write('')
st.write('')

date_list, thermique_list, eolien_list, solaire_list, hydraulique_list, bioenergies_list, ech_physiques_list = knn_production(df_train, X_test, predictions, str(date1),20)

x = date_list

layout2 = go.Layout(yaxis = dict(title = 'Production (MW)',zeroline=True,showline = True),xaxis = dict(title = 'Date', zeroline = True,showline = True)
                    ,legend=dict(yanchor="top",y=0.99,xanchor="left",x=1,font=dict(family="Roboto",size=10,color="black")))

#Prediction graph of energy production
fig4 = plt.figure(figsize=(10, 4))
fig4 = go.Figure(go.Bar(x=x, y= hydraulique_list, name='Hydraulique'),layout=layout2)
fig4.add_trace(go.Bar(x=x, y= eolien_list, name='Eolien'))
fig4.add_trace(go.Bar(x=x, y= solaire_list, name='Solaire'))
fig4.add_trace(go.Bar(x=x, y= bioenergies_list, name='Bioenergies'))
fig4.add_trace(go.Bar(x=x, y= thermique_list, name='Thermique'))
fig4.add_trace(go.Bar(x=x, y= ech_physiques_list, name="importations d'électricité"))
fig4.update_layout(barmode='stack', xaxis={'categoryorder':'total descending'})
fig4.update_layout(title_text=f"Prévision de répartition de production<br>entre le {date1} et le {date2}", title_x=0.5,title_y=0.9, font=dict(family="Roboto",size=14,color="black"))#,fontsize = 8)
st.plotly_chart(fig4)

