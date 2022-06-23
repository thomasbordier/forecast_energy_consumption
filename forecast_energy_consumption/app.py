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

#TODO
#remonter input date 1 year, graph,
# camebert production
# prediction 14 jours
# production thomas

'''
# Energy consumption forecast
'''

#st.markdown("<h1 style='text-align: center; color: black;'>Energy consumption forecast</h1>", unsafe_allow_html=True)

#face = image.imread("image_test.png")
#image = Image.open('image_test.jpeg')
#st.image(image, width=1200)

###
#col1, col2, col3 = st.columns([1,6,1])

#with col1:
#    st.write("")

#with col2:
#    st.image(image, caption='DEMO of the output', width=1200)

#with col3:
#    st.write("")

###

'''
### Query period forecast energy consumption :
'''

url = 'http://127.0.0.1:5000/predict' #'https://taxifare.lewagon.ai/predict' (exercice 1, jeudi)


date1 = st.date_input(label= "Starting date :", value= datetime.date(2020, 1, 1), min_value=datetime.date(2020, 1, 1), max_value=datetime.date(2022, 4, 30))

date2 = date1 + timedelta(days = 13)

date_test = pd.DataFrame([date1,date2]).set_index(0).asfreq('D') 

#date2 = st.date_input(label= "Ending date :", value= datetime.date(2022, 4, 30), min_value=datetime.date(2013, 1, 1), max_value=datetime.date(2022, 4, 30))

st.write('The energy consumption forecast from',date1,'to',date2,'is :')

#date_test = pd.DataFrame(['2015-01-01','2015-01-05']).set_index(0).asfreq('D')
date_test = pd.DataFrame([date1,date2]).set_index(0).asfreq('D')

#X_train,y_train,X_test,y_test,df_train = X_y_train_test(str(date1), 14)


df_train, X_test, y_test, predictions, mape = main('xgb',str(date1), 14)

date_train, prod_history = consumption_history(str(date1))

st.markdown("<h2 style='text-align: center; color: black;'>Energy consumption forecast 1 year before</h2>", unsafe_allow_html=True)
fig1 = plt.figure(figsize=(10, 4))
fig1 = px.line(date_train, x="Date", y="Consommation (MW)", title='Energy consumption 1 year in PACA before')#,title_x = 5)#,title_size = 10)

import textwrap
split_text = textwrap.wrap('This is a very long title and it would be great to have it on three lines', 
                            width=30)

fig1.update_layout(title_text=f"Consommation d'énergie \n entre le {date1} et le {date2}", title_x=0.5,title_y=0.85, font=dict(family="Courier New, monospace",size=18,color="RebeccaPurple"))#,fontsize = 10)
#fig1.update_layout(title_text=split_text, title_x=0.5,title_y=0.85, font=dict(family="Courier New, monospace",size=18,color="RebeccaPurple"))#,fontsize = 10)

st.plotly_chart(fig1)

RPA = prod_history

fig2 = px.pie(values = np.array(prod_history.values).tolist()[0],names = prod_history.columns)

st.plotly_chart(fig2)

st.write('The energy consumption forecast from',date1,'to',date2,'is :')

fig3 = plt.figure(figsize=(10, 4))
fig3 = px.line(x=date_test.index, y=predictions)#, title='Energy consumption forecast for the 14 next days')
st.plotly_chart(fig3)

st.write('erreur moyenne:',round(mape,4),'%')

date_list, thermique_list, eolien_list, solaire_list, hydraulique_list, bioenergies_list, ech_physiques_list = knn_production(df_train, X_test, predictions, str(date1),20)

x = date_list

fig4 = go.Figure(go.Bar(x=x, y= hydraulique_list, name='Hydraulique'))
fig4.add_trace(go.Bar(x=x, y= eolien_list, name='Eolien'))
fig4.add_trace(go.Bar(x=x, y= solaire_list, name='Solaire'))
fig4.add_trace(go.Bar(x=x, y= bioenergies_list, name='Bioenergies'))
fig4.add_trace(go.Bar(x=x, y= thermique_list, name='Thermique'))
fig4.add_trace(go.Bar(x=x, y= ech_physiques_list, name='Echanges physiques'))
fig4.update_layout(barmode='stack', xaxis={'categoryorder':'total descending'})
st.plotly_chart(fig4)

