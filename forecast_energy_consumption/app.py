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


date1 = st.date_input(label= "Starting date :", value= datetime.date(2015, 1, 1), min_value=datetime.date(2013, 1, 1), max_value=datetime.date(2022, 4, 30))

date2 = date1 + timedelta(days = 13) 

#date2 = st.date_input(label= "Ending date :", value= datetime.date(2022, 4, 30), min_value=datetime.date(2013, 1, 1), max_value=datetime.date(2022, 4, 30))

st.write('The energy consumption forecast from',date1,'to',date2,'is :')

#date_test = pd.DataFrame(['2015-01-01','2015-01-05']).set_index(0).asfreq('D')
date_test = pd.DataFrame([date1,date2]).set_index(0).asfreq('D') 
    
#X_train,y_train,X_test,y_test,df_train = X_y_train_test(str(date1), 14)

df_train, X_test, y_test, predictions, mape = main('xgb',date1, 14)

#y_pred, mape_pred = predict_output(X_test,y_test, metric = True)

date_train, prod_history = consumption_history(str(date1))


############### One year before
#fig = plt.figure(figsize=(10, 4))
#plt.plot(pd.to_datetime(date_train['Date']),date_train['Consommation (MW)'], label="y_true")
#plt.title("Energy consumption 1 year before",fontsize=14, fontweight='bold')
#plt.legend()
#st.pyplot(fig1)

#df = px.data.gapminder().query("country=='Canada'")
fig1 = plt.figure(figsize=(10, 4))
fig1 = px.line(date_train, x="Date", y="Consommation (MW)", title='Energy consumption 1 year before')
st.plotly_chart(fig1)
#fig1.show()

################

RPA = prod_history

fig = px.pie(values = np.array(prod_history.values).tolist()[0],names = prod_history.columns)
#fig.show()
st.plotly_chart(fig)



################ GR

fig2 = plt.figure(figsize=(10, 4))
fig2 = px.line(x=date_test.index, y=predictions, title='Energy consumption forecast for the 14 next days')
st.plotly_chart(fig2)

################

st.write('erreur moyenne:',round(mape,4),'%')

#knn_production(df_train, X_test, y_pred, Date_debut_test)
date_list, thermique_list, eolien_list, solaire_list, hydraulique_list, bioenergies_list, ech_physiques_list = knn_production(df_train, X_test, y_pred, date1)
#fig = px.line(df_energy_weather[['Consommation (MW)','Ech. physiques (MW)']]['2018':'2020'].resample('M').mean(),title='Evolution des échanges physiques par rapport à nos besoins en énergie entre 2018 et 2020')
