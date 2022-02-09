import pandas as pd
import numpy as np
import json

import requests

import folium
import streamlit as st
from streamlit_folium import folium_static

add_select = st.sidebar.selectbox("What data do you want to see?", ("OpenStreetMap", "Stamen Terrain", "Stamen Toner"))
week_select = st.sidebar.selectbox("Choose a week", ("Christmas", "Thanksgiving", "Halloween"))

chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['a', 'b', 'c'])

st.line_chart(chart_data)

map_sby = folium.Map(tiles=add_select, location=[40.730610, -73.935242], zoom_start=12)

st.title('Map of Surabaya')
folium_static(map_sby)
