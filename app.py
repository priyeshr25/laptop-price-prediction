import numpy as np
import streamlit as st
import pickle
import sklearn
import pandas as pd

pipe=pickle.load(open('model_.pkl','rb'))
df=pickle.load(open('data.pkl','rb'))

st.title('Laptop predictor')

brand=st.selectbox('Brand',df['brand'].unique())

processor_name=st.selectbox('Processor Name',df['processor_name'].unique())

ram_expandable=st.selectbox('Ram Expandable',df['ram_expandable'].unique())

ram=st.selectbox('Ram (in GB)',[2,4,6,8,12,16,32,64])

ram_type=st.selectbox('Ram type',df['ram_type'].unique())

ghz=st.selectbox('GHZ',df['ghz'].unique())

display_type=st.selectbox('Display Type',df['display_type'].unique())

display=st.selectbox('Display Type',df['display'].unique())

gpu=st.selectbox('gpu',df['gpu'].unique())

gpu_brand=st.selectbox('Gpu Brand',df['gpu_brand'].unique())

ssd=st.selectbox('SSD',df['ssd'].unique())

hdd=st.selectbox('HDD',df['hdd'].unique())

adapter=st.selectbox('Adapter',df['adapter'].unique())

battery_life=st.selectbox('Battery Life(hours)',df['battery_life'].unique())

brand_name=st.selectbox('Brand Name',df['brand_name'].unique())

windows=st.selectbox('Windows',df['windows'].unique())

if st.button('Predict Price'):
    query_df = pd.DataFrame([[brand, processor_name, ram_expandable, ram, ram_type, ghz,
                              display_type, display, gpu, gpu_brand, ssd, hdd, adapter,
                              battery_life, brand_name, windows]],
                            columns=['brand', 'processor_name', 'ram_expandable', 'ram', 'ram_type', 'ghz',
                                     'display_type', 'display', 'gpu', 'gpu_brand', 'ssd', 'hdd', 'adapter',
                                     'battery_life', 'brand_name',
                                     'windows'])  # Ensure column order matches model training

    # Predict and display the result



    prediction = pipe.predict(query_df)
    st.title(f"Predicted Price: ₹{prediction[0]:,.2f}")