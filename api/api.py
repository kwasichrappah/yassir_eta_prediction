from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
from pydantic import BaseModel
from datetime import datetime

data_cleaner = joblib.load("../models/cleaner.joblib")
dtree_pipeline = joblib.load("../models/D_Tree.joblib")
linear_pipeline = joblib.load("../models/Linear_Regression.joblib")
svr_pipeline = joblib.load("../models/SVR.joblib")
xgboost_pipeline = joblib.load("../models/Xgboost.joblib")

w_df = pd.read_csv("data/Weather_data.csv",parse_dates = ['date'])

'''to run the API, run this line which is based on the api directory then 'uvicorn api(api python file):app(instance of fast API) --reload' '''
# Create a FastAPI application
app = FastAPI()

# class patient_features(BaseModel):
# 	ID : str
# 	Timestamp : datetime
# 	Origin_lat :float
# 	Origin_lon :float
# 	Destination_lat :float
# 	Destination_lon :float
# 	Trip_distance :float
	
       
# Define a route at the root web address ("/")
@app.get("/")
def status_check():
	return {"Status": "API is online!!!"}




# @app.post("/xgboost_model")
# def predict_sepssis(data:patient_features):

#     df = pd.DataFrame([data.model_dump()])
#     cleaned = data_cleaner(df,w_df)
#     forecast=xgboost_pipeline.predict(cleaned)
#     prediction = np.expm1(forecast)

 
#     return {"prediction": prediction}


# @app.post("/log_model")
# def predict_sepssis(data:patient_features):

#     df = pd.DataFrame([data.model_dump()])
#     prediction=log_pipeline.predict(df)
#     prediction = int(prediction[0])
#     probability = xgb_pipeline.predict_proba(df)

#     prediction = encoder.inverse_transform([prediction])[0]
 
#     if prediction == 'Negative':
#             probability= f'{round(probability[0][0], 2)*100}%'
#     else:
#             probability = f'{round(probability[0][1], 2)*100}%'
 
#     return {"prediction": prediction, "probability": probability}


# @app.post("/svc_model")
# def predict_sepssis(data:patient_features):

#     df = pd.DataFrame([data.model_dump()])
#     prediction=svc_pipeline.predict(df)
#     prediction = int(prediction[0])
#     probability = xgb_pipeline.predict_proba(df)

#     prediction = encoder.inverse_transform([prediction])[0]
 
#     if prediction == 'Negative':
#             probability= f'{round(probability[0][0], 2)*100}%'
#     else:
#             probability = f'{round(probability[0][1], 2)*100}%'
 
#     return {"prediction": prediction, "probability": probability}

# @app.post("/catboost_model")
# def predict_sepssis(data:patient_features):

#     df = pd.DataFrame([data.model_dump()])
#     prediction=catboost_pipeline.predict(df)
#     prediction = int(prediction[0])
#     probability = xgb_pipeline.predict_proba(df)

#     prediction = encoder.inverse_transform([prediction])[0]
 
#     if prediction == 'Negative':
#             probability= f'{round(probability[0][0], 2)*100}%'
#     else:
#             probability = f'{round(probability[0][1], 2)*100}%'
 
#     return {"prediction": prediction, "probability": probability}