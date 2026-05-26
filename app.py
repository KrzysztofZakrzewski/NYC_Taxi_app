# 🚕 NYC Taxi Trip Duration — Streamlit App

import streamlit as st
import pandas as pd
import numpy as np
import joblib

from math import radians, sin, cos, sqrt, atan2, degrees


# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title='NYC Taxi Trip Duration Predictor',
    page_icon='🚕',
    layout='centered'
)


# =====================================
# LOAD MODEL
# =====================================

model = joblib.load('models/random_forest_model.pkl')


# =====================================
# DISTANCE FUNCTION
# =====================================


def calculate_distance(
    pickup_latitude,
    pickup_longitude,
    dropoff_latitude,
    dropoff_longitude
):

    R = 6371

    lat1 = radians(pickup_latitude)
    lon1 = radians(pickup_longitude)

    lat2 = radians(dropoff_latitude)
    lon2 = radians(dropoff_longitude)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        sin(dlat / 2) ** 2
        + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    )

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c


# =====================================
# BEARING FUNCTION
# =====================================


def calculate_bearing(
    pickup_latitude,
    pickup_longitude,
    dropoff_latitude,
    dropoff_longitude
):

    lat1 = radians(pickup_latitude)
    lat2 = radians(dropoff_latitude)

    diff_long = radians(
        dropoff_longitude - pickup_longitude
    )

    x = sin(diff_long) * cos(lat2)

    y = (
        cos(lat1) * sin(lat2)
        - sin(lat1) * cos(lat2) * cos(diff_long)
    )

    initial_bearing = atan2(x, y)

    initial_bearing = degrees(initial_bearing)

    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing


# =====================================
# TITLE
# =====================================

st.title('🚕 NYC Taxi Trip Duration Predictor')

st.markdown(
    'Predict NYC taxi trip duration using a trained Random Forest regression model.'
)


# =====================================
# USER INPUTS
# =====================================

st.header('📍 Trip Information')

vendor_id = st.selectbox(
    'Vendor ID',
    [1, 2]
)

passenger_count = st.slider(
    'Passenger Count',
    min_value=1,
    max_value=6,
    value=1
)

store_and_fwd_flag = st.selectbox(
    'Store and Forward Flag',
    [0, 1]
)


st.header('🗺️ Pickup Coordinates')

pickup_latitude = st.number_input(
    'Pickup Latitude',
    value=40.740757,
    format='%.6f'
)

pickup_longitude = st.number_input(
    'Pickup Longitude',
    value=-73.990122,
    format='%.6f'
)


st.header('🏁 Dropoff Coordinates')

dropoff_latitude = st.number_input(
    'Dropoff Latitude',
    value=40.701203,
    format='%.6f'
)

dropoff_longitude = st.number_input(
    'Dropoff Longitude',
    value=-74.013972,
    format='%.6f'
)


st.header('⏰ Temporal Features')

pickup_hour = st.slider(
    'Pickup Hour',
    min_value=0,
    max_value=23,
    value=10
)

pickup_weekday = st.slider(
    'Pickup Weekday (0 = Monday)',
    min_value=0,
    max_value=6,
    value=2
)

pickup_month = st.slider(
    'Pickup Month',
    min_value=1,
    max_value=12,
    value=5
)


# =====================================
# PREDICTION
# =====================================

if st.button('🚀 Predict Trip Duration'):

    distance_km = calculate_distance(
        pickup_latitude,
        pickup_longitude,
        dropoff_latitude,
        dropoff_longitude
    )

    bearing = calculate_bearing(
        pickup_latitude,
        pickup_longitude,
        dropoff_latitude,
        dropoff_longitude
    )

    is_weekend = 1 if pickup_weekday >= 5 else 0

    hour_sin = np.sin(
        2 * np.pi * pickup_hour / 24
    )

    hour_cos = np.cos(
        2 * np.pi * pickup_hour / 24
    )

    sample_trip = pd.DataFrame([{
        'vendor_id': vendor_id,
        'passenger_count': passenger_count,

        'pickup_longitude': pickup_longitude,
        'pickup_latitude': pickup_latitude,

        'dropoff_longitude': dropoff_longitude,
        'dropoff_latitude': dropoff_latitude,

        'store_and_fwd_flag': store_and_fwd_flag,

        'distance_km': distance_km,

        'pickup_hour': pickup_hour,
        'pickup_weekday': pickup_weekday,
        'is_weekend': is_weekend,
        'pickup_month': pickup_month,

        'hour_sin': hour_sin,
        'hour_cos': hour_cos,

        'bearing': bearing
    }])

    prediction = float(model.predict(sample_trip)[0])

    st.success('Prediction completed successfully!')

    st.metric(
        label='Predicted Trip Duration (Seconds)',
        value=f'{prediction:.2f}'
    )

    st.metric(
        label='Predicted Trip Duration (Minutes)',
        value=f'{prediction / 60:.2f}'
    )

    st.markdown('---')

    st.subheader('📊 Auto-Generated Features')

    st.write(f'Distance (km): {distance_km:.2f}')
    st.write(f'Bearing: {bearing:.2f}')
    st.write(f'Weekend Trip: {is_weekend}')