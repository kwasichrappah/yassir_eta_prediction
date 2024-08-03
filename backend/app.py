from fastapi import FastAPI
import pandas as pd
import joblib
import numpy as np
from pydantic import BaseModel
from datetime import datetime
from scipy.ndimage import gaussian_filter1d


def cleaner(df,w_df):
    df['date'] = df['Timestamp'].dt.date
    df = df.sort_values(by='Timestamp')
    df['date'] = pd.to_datetime(df['date'])
    w_df['date'] = pd.to_datetime(w_df['date'])

    # Merge DataFrames on 'date' column

    merged_df = pd.merge(df, w_df, on='date', how='left')
    merged_df['day']= merged_df['date'].dt.day_name()
    merged_df['daynumber']= merged_df['date'].dt.day
    merged_df['hour']= merged_df['Timestamp'].dt.hour

    # Calculating wind speed
    merged_df['wind_sqrt'] = np.sqrt((merged_df['u_component_of_wind_10m'] ** 2) + (merged_df['v_component_of_wind_10m'] ** 2))
    # Apply log transformations
    # merged_df['log_eta'] = np.log1p(merged_df['ETA'])
    merged_df['log_mean_2m_air_temperature'] = np.log1p(merged_df['mean_2m_air_temperature'])

    #Apply cube root transformations
    merged_df['log_Trip_distance'] = np.cbrt(merged_df['Trip_distance'])

    # Apply inverse log transformation
    merged_df['log_total_precipitation'] = np.expm1(merged_df['total_precipitation'])


    #Smoothening some columns

    # Apply Gaussian filter with a sigma of 1
    merged_df['gaussian_surface_pressure'] = gaussian_filter1d(merged_df['surface_pressure'], sigma=1)
    merged_df['gaussian_mean_sea_level'] = gaussian_filter1d(merged_df['mean_sea_level_pressure'], sigma=1)
    merged_df['gaussian_mean_total_precipitation'] = gaussian_filter1d(merged_df['log_total_precipitation'], sigma=4)
    merged_df['gaussian_dewpoint_2m_temperature'] = gaussian_filter1d(merged_df['dewpoint_2m_temperature'], sigma=1)

    merged_df.drop(columns=['Origin_lat','Origin_lon','Destination_lat','Destination_lon','ID','date','daynumber','Trip_distance','Timestamp',
                              'maximum_2m_air_temperature','mean_2m_air_temperature','mean_sea_level_pressure','dewpoint_2m_temperature',
                              'day','minimum_2m_air_temperature','surface_pressure','u_component_of_wind_10m','v_component_of_wind_10m',
                              'gaussian_mean_total_precipitation','ETA','wind_sqrt','total_precipitation','log_total_precipitation'],axis=1,inplace=True)
    return merged_df	

w_df = pd.read_csv("../data/Weather_data.csv",parse_dates = ['date'])
#data_cleaner = joblib.load("../models/cleaner.joblib")
dtree_pipeline = joblib.load("../models/D_Tree.joblib")
linear_pipeline = joblib.load("../models/Linear_Regression.joblib")
svr_pipeline = joblib.load("../models/SVR.joblib")
xgboost_pipeline = joblib.load("../models/Xgboost.joblib")
# Create a FastAPI application
app = FastAPI()

class ride_order_features(BaseModel):
	ID : str
	Timestamp : datetime
	Origin_lat :float
	Origin_lon :float
	Destination_lat :float
	Destination_lon :float
	Trip_distance :float

# Define a route at the root web address ("/")
@app.get("/")
def status_check():
	return {"Status": "Yassir ETA Predictor API is online!!!"}


@app.post("/xgb")
def predict_sepssis(data:ride_order_features):

    df = pd.DataFrame([data.model_dump()])
    cleaned = cleaner(df,w_df)
    forecast=xgboost_pipeline.predict(cleaned)
    prediction = np.expm1(forecast)

 
    return {"prediction": prediction}

@app.post("/dtree")
def predict_sepssis(data:ride_order_features):

    df = pd.DataFrame([data.model_dump()])
    cleaned = cleaner(df,w_df)
    forecast=dtree_pipeline.predict(cleaned)
    prediction = np.expm1(forecast)

 
    return {"prediction": prediction}

@app.post("/linear")
def predict_sepssis(data:ride_order_features):

    df = pd.DataFrame([data.model_dump()])
    cleaned = cleaner(df,w_df)
    forecast=linear_pipeline.predict(cleaned)
    prediction = np.expm1(forecast)

 
    return {"prediction": prediction}