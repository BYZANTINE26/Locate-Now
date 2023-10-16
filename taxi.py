import streamlit as st
import pickle
import numpy as np
import folium
from streamlit_folium import folium_static
import pandas as pd
from datetime import datetime, timedelta

import asyncio
import subprocess

def predict(x):
    with open('xg_model.pkl', 'rb') as file:
        model = pickle.load(file)
        
    y_pred = model.predict(x)
    return y_pred

def main():
    
    page_bg = """
    <style>
    body {
    background-color: #F5F5F5;
    }
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)

    st.image('logo.png')

    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")


    latest_data = pd.read_csv('updated_data.csv')

    x_latest_data = latest_data.values[:, :]  

    y_pred = predict(x_latest_data)
    xgb_test_predictions = [round(value) for value in y_pred]
    

    m = folium.Map(location=[40.7128, -74.0060], zoom_start=10)

    for i, point in enumerate(y_pred):
        lat = latest_data.iloc[i]['lat']
        lon = latest_data.iloc[i]['lon']
        folium.map.Marker(
            location=[lat, lon],
            icon=folium.Icon(color='red'),
            tooltip='<h1 style="font-size: 20px;">{}</h1>'.format(round(point)),  
        ).add_to(m)

    folium_static(m)
    
    st.write('Updated time',  current_time)

if __name__ == '__main__':
    main()

