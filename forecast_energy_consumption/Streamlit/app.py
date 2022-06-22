import streamlit as st
import datetime
import matplotlib.pyplot as plt
from matplotlib import image
from scipy import misc
from PIL import Image



'''
# Energy consumption forecast 
'''

#st.markdown("<h1 style='text-align: center; color: black;'>Energy consumption forecast</h1>", unsafe_allow_html=True)

#face = image.imread("image_test.png")
image = Image.open('image_test.jpeg')
st.image(image, width=1200)


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


date1 = st.date_input(label= "Starting date :", value= datetime.date(2013, 1, 1), min_value=datetime.date(2013, 1, 1), max_value=datetime.date(2022, 4, 30))

date2 = st.date_input(label= "Ending date :", value= datetime.date(2022, 4, 30), min_value=datetime.date(2013, 1, 1), max_value=datetime.date(2022, 4, 30))

st.write('The energy consumption forecast from',date1,'to',date2,'is :')

#if date2 > datetime.date(2022, 4, 30):
#    st.error('The ending date should not be higher than 2022/04/30')

''''
latitude_pickup = st.number_input('Insert the latitude_pickup')
st.write('The pickup latitude is ', latitude_pickup)

longitude_pickup = st.number_input('Insert the longitude_pickup')
st.write('The pickup longitude is ', longitude_pickup)

latitude_pickup = st.number_input('Insert the latitude_pickup')
st.write('The pickup latitude is ', latitude_pickup)

longitude_pickup = st.number_input('Insert the longitude_pickup')
st.write('The pickup longitude is ', longitude_pickup)

#DICT = {}
'''