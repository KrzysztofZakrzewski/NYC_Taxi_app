# 🚕 NYC Taxi Trip Duration — Streamlit App

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pydeck as pdk

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

st.markdown(
    "<h1 style='color:#F4A300;'>🚕 NYC Taxi Predictor</h1>",
    unsafe_allow_html=True
)

st.markdown(
    'Predict NYC taxi trip duration using a trained Random Forest regression model.'
)


with st.expander("📊 Show Auto-Generated Features"):
    
    st.write(
        """
        The model automatically generates the following features based on the input coordinates and temporal information:
        
        - **Distance (km)**: The Haversine distance between pickup and dropoff points.
        - **Bearing**: The compass bearing from pickup to dropoff location.
        - **Weekend Trip**: A binary feature indicating if the trip starts on a weekend.
        - **Hour Sin/Cos**: Sine and cosine transformations of the pickup hour to capture cyclical patterns.
        """
    )


places = pd.DataFrame({
    'name': [
        'Flatiron Building',
        'Times Square',
        'Grand Central Terminal',
        'Brooklyn Bridge',
        'Coney Island',
        'Barclays Center',
        'John F. Kennedy International Airport',
        'LaGuardia Airport',
        'Yankee Stadium',
        'Arthur Ashe Stadium',
        'One World Trade Center',
        'Rockefeller Center',
        'Bronx Zoo',
        'Prospect Park',
        'Columbia University'
    ],

    'lat': [
        40.741112,
        40.758896,
        40.752726,
        40.706086,
        40.574926,
        40.682650,
        40.641311,
        40.776927,
        40.829643,
        40.749824,
        40.712743,
        40.758740,
        40.850596,
        40.660204,
        40.807536
    ],

    'lon': [
        -73.989723,
        -73.985130,
        -73.977229,
        -73.996864,
        -73.985941,
        -73.975280,
        -73.778139,
        -73.873966,
        -73.926175,
        -73.845836,
        -74.013379,
        -73.978674,
        -73.876998,
        -73.968956,
        -73.962573
    ]
})

with st.expander("🗺️ Common Places in NYC MAP"):
    # st.map(places)

    st.pydeck_chart(
        pdk.Deck(

            map_style='light',

            initial_view_state=pdk.ViewState(
                latitude=40.758896,
                longitude=-73.985130,
                zoom=11,
                pitch=0,
            ),

            layers=[
                pdk.Layer(
                    'ScatterplotLayer',
                    data=places,

                    get_position='[lon, lat]',

                    get_radius=100,

                    get_fill_color='[255, 165, 0]',

                    pickable=True,
                )
            ],

            tooltip={
                'html': '<b>{name}</b><br/>Lat: {lat}<br/>Lon: {lon}',
                'style': {
                    'backgroundColor': 'white',
                    'color': 'black'
                }
            }
        )
    )

# =====================================
# LOAD MODEL
# =====================================

model_version = st.radio(
    'Select Model Version',
    ['200K Optimized Model', '1.4M Full Model']
)

if model_version == '1.4M Full Model':
    model = joblib.load(
        'models/random_forest_model.pkl'
    )
else:
    model = joblib.load(
        'models/random_forest_model_sample.pkl'
    )

# =====================================
# USER INPUTS
# =====================================

st.markdown(
    "<h2 style='color:#003366;'>🍎️ Trip Information</h2>",
    unsafe_allow_html=True
)

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

st.markdown(
    "<h2 style='color:#003366;'>🗺️ Pickup Coordinates</h2>",
    unsafe_allow_html=True
)

pickup_place = st.selectbox(
    'Select Pickup Location',
    places['name'],
    index=0
)

pickup_selected = places[
    places['name'] == pickup_place
].iloc[0]

pickup_latitude = st.number_input(
    'Pickup Latitude',
    value=float(pickup_selected['lat']),
    format='%.6f'
)

pickup_longitude = st.number_input(
    'Pickup Longitude',
    value=float(pickup_selected['lon']),
    format='%.6f'
)

st.markdown(
    "<h2 style='color:#003366;'>🏁 Dropoff Coordinates</h2>",
    unsafe_allow_html=True
)

dropoff_place = st.selectbox(
    'Select Dropoff Location',
    places['name'],
    index=1
)


dropoff_selected = places[
    places['name'] == dropoff_place
].iloc[0]

dropoff_latitude = st.number_input(
    'Dropoff Latitude',
    value=float(dropoff_selected['lat']),
    format='%.6f'
)

dropoff_longitude = st.number_input(
    'Dropoff Longitude',
    value=float(dropoff_selected['lon']),
    format='%.6f'
)


st.markdown(
    "<h2 style='color:#003366;'>🍊️ Temporal Features</h2>",
    unsafe_allow_html=True
)

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

if st.button('⌚️ Predict Trip Duration'):

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

with st.expander("👨‍💻 Credits & Resources"):

    st.markdown(
        """
        ### 🚕 NYC Taxi Trip Duration Predictor

        **Author:**  
        Krzysztof Zakrzewski

        ---

        🔗 **Portfolio**  
        https://krzysztofzakrzewski.github.io/portfolio/

        🔗 **LinkedIn**  
        https://www.linkedin.com/in/TWOJ_LINKEDIN/

        🔗 **GitHub Repositories**
        - EDA Project  
          https://github.com/KrzysztofZakrzewski/NYC_Taxi_Trip_Duration_EDA

        - ML Project  
          https://github.com/KrzysztofZakrzewski/NYC_Taxi_Trip_Duration_ML

        - Streamlit App  
          https://github.com/KrzysztofZakrzewski/NYC_Taxi_app

        ---

        📂 **Dataset Source (Kaggle)**  
        https://www.kaggle.com/datasets/yasserh/nyc-taxi-trip-duration

        🏛️ **Original Data Provider**  
        NYC Taxi and Limousine Commission (TLC)

        🔗 **Official TLC Data Portal**  
        https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
        """
    )