import pickle
import pandas as pd  # Add this at the top if not already imported
import numpy as np
import streamlit as st

# import the model
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# st.write("Columns in DataFrame 'df':")
# st.write(df.columns)
# st.write("First 5 rows of DataFrame 'df':")
# st.write(df.head())

st.title("Laptop Predictor")

# brand
company = st.selectbox('Brand', df['Company'].unique())

# type of laptop
Type = st.selectbox('Type', df['TypeName'].unique())

# Ram
Ram = st.selectbox('RAM(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# weight
weight = st.number_input('Weight of the Laptop')

# Touchscreen
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

# IPS
ips = st.selectbox('IPS', ['No', 'Yes'])

# screen size
screen_size = st.number_input('Inches')

# resolution
resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768',
                                                '1600x900', '3840x2160',
                                                '3200x1800', '2880x1800',
                                                '2560x1600', '2560x1440',
                                                '2304x1440'])

# CPU
cpu = st.selectbox('CPU', df['Cpu brand'].unique())

hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])

gpu = st.selectbox('GPU', df['Gpu brand'].unique())

os = st.selectbox('OS', df['os'].unique())

if st.button('Predict Price'):
    ppi = None
    if touchscreen != "Yes":
        touchscreen = 0
    else:
        touchscreen = 1

    if ips == "Yes":
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split("x")[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

    # query = np.array([company, Type, Ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])
    # query = query.reshape(1, 12)
    # st.title("The Predicted Price of Laptop = Rs " + str(int(np.exp(pipe.predict(query)[0]))))

    query = pd.DataFrame([[company, Type, Ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os]],
                         columns=['Company', 'TypeName', 'Ram', 'Weight', 'Touchscreen', 'Ips',
                                  'ppi', 'Cpu brand', 'HDD', 'SSD', 'Gpu brand', 'os'])

    # Optional: Force string columns to string type to avoid Unicode errors
    for col in query.select_dtypes(include='object').columns:
        query[col] = query[col].astype(str)

    # Prediction
    st.title("The Predicted Price of Laptop = Rs " + str(int(np.exp(pipe.predict(query)[0]))))
