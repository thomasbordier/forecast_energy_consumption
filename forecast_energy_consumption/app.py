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

st.write(type(str(date1)))

#date2 = st.date_input(label= "Ending date :", value= datetime.date(2022, 4, 30), min_value=datetime.date(2013, 1, 1), max_value=datetime.date(2022, 4, 30))

st.write('The energy consumption forecast from',date1,'to',date2,'is :')

#date_test = pd.DataFrame(['2015-01-01','2015-01-05']).set_index(0).asfreq('D')
date_test = pd.DataFrame([date1,date2]).set_index(0).asfreq('D') 
    
X_train,y_train,X_test,y_test = X_y_train_test(str(date1), 14)

y_pred = predict_output(X_test,y_test, metric = True)

df = pd.DataFrame(y_pred,columns=['y_pred'])
df['Date'] = date_test.index
df = df.set_index('Date')
st.line_chart(df)

#date_train, prod_history = consumption_history(X_train)

import matplotlib.pyplot as plt
import numpy as np


ax, fig = plt.plot(date_test.index,y_pred, label="x**2")

st.pyplot(fig)

