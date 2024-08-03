from fastapi import FastAPI
import pandas as pd
import joblib
import numpy as np
from pydantic import BaseModel
from datetime import datetime


w_df = pd.read_csv("../data/Weather_data.csv",parse_dates = ['date'])
data_cleaner = joblib.load("../models/cleaner.joblib")
dtree_pipeline = joblib.load("../models/D_Tree.joblib")
linear_pipeline = joblib.load("../models/Linear_Regression.joblib")
svr_pipeline = joblib.load("../models/SVR.joblib")
xgboost_pipeline = joblib.load("../models/Xgboost.joblib")
# Create a FastAPI application
app = FastAPI()

# Define a route at the root web address ("/")
@app.get("/")
def status_check():
	return {"Status": "Yassir ETA Predictor API is online!!!"}
