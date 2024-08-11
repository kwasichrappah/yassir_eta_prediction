import streamlit as st
import pandas as pd
import joblib
from scipy.ndimage import gaussian_filter1d
import os
import datetime
import numpy as np
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

#setting page title and icon
st.set_page_config(
    page_title = "Prediction Page",
    page_icon = "🎯",
    layout = 'wide'
)

w_df = pd.read_csv('data/Weather_data.csv')

#Loading the models into streamlit app
st.cache_resource(show_spinner="Models Loading")
def load_dtree_pipeline():
    pipeline = joblib.load("./models/D_Tree.joblib")
    return pipeline


st.cache_resource(show_spinner="Models Loading")
def load_linear_regressor_pipeline():
    pipeline = joblib.load('./models/Linear_Regression.joblib')
    return pipeline


st.cache_resource(show_spinner="Models Loading")
def load_svr_pipeline():
    pipeline = joblib.load('./models/SVR.joblib')
    return pipeline


st.cache_resource(show_spinner="Models Loading")
def load_xgboost_pipeline():
    pipeline = joblib.load('./models/xgboost_tuned.joblib')
    return pipeline

#Selecting model for prediction
def select_model():
        col1,col2 = st.columns(2)

        with col2:
             st.selectbox('Select a Model', options = ['D-Tree','Linear Regressor','XGBoost','SVR'],key='selected_model')

        if st.session_state['selected_model'] == 'D-Tree':
             pipeline = load_dtree_pipeline()
        
        elif st.session_state['selected_model'] == 'Linear Regressor':
             pipeline = load_linear_regressor_pipeline()

        elif st.session_state['selected_model'] == 'XGBoost':
             pipeline = load_xgboost_pipeline()
        else:
             pipeline = load_svr_pipeline()

        return pipeline


def cleaner(df):
   #Apply cube root transformations
    df['log_Trip_distance'] = np.log1p(df['tripdistance'])

    #Smoothening some columns
    df['log_mean_2m_air_temperature'] = np.log1p(df['mean_2m_air_temperature'])

    # Apply Gaussian filter with a sigma of 1
    df['gaussian_surface_pressure'] = gaussian_filter1d(np.log1p(df['surface_pressure']), sigma=1)
    df['gaussian_mean_sea_level'] = gaussian_filter1d(np.log1p(df['mean_sea_level_pressure']), sigma=1)
    df['gaussian_dewpoint_2m_temperature'] = gaussian_filter1d(np.log1p(df['dewpoint_2m_temperature']), sigma=1)

    
    columns=['log_mean_2m_air_temperature','gaussian_surface_pressure','gaussian_dewpoint_2m_temperature','gaussian_mean_sea_level','log_Trip_distance']

    # Select only columns 'A' and 'C' using .filter
    df= df.filter(columns)
    df= df.reindex(columns=['log_mean_2m_air_temperature','gaussian_surface_pressure','gaussian_dewpoint_2m_temperature','gaussian_mean_sea_level','log_Trip_distance'])

    return df


#Prediction and probability variables state at the start of the webapp
if 'prediction' not in st.session_state:
     st.session_state['prediction'] = None
     st.session_state['id'] = None


#Make Prediction on large data
def large_prediction():
     st.markdown("### Predicting many Trips")
     pipeline = select_model()
     with st.expander("Click here to preview uploaded data"):
          st.write("Yassir Ride Orders")
          df = pd.read_csv(uploaded_file)
          st.write(df.head()) 
    
     if st.button('Yassir Predict 👍'):
     
          df.Timestamp = pd.to_datetime(df.Timestamp)
          df['date'] = df['Timestamp'].dt.date
          df['time'] = df['Timestamp'].dt.time
          w_df['date'] = pd.to_datetime(w_df['date']).dt.date
          merged_df = pd.merge(df, w_df, on='date', how='left')
          merged_df.rename(columns={'Trip_distance': 'tripdistance'}, inplace=True)
          merged_df = cleaner(merged_df)
          pred = pipeline.predict(merged_df)
          prediction = np.expm1(pred)
          merged_df['ID'] = df['ID']
          merged_df['tripdistance'] = df['Trip_distance']
          merged_df['Pred_ETA']= prediction

          merged_df[['ID','tripdistance','Pred_ETA']].to_csv('.\\data\\history.csv',mode='a',header = not os.path.exists('.\\data\\history.csv'),index=False)
          st.write(merged_df[['ID','Pred_ETA']])


#Making prediction 
def make_prediction(pipeline):
     id = st.session_state['id']
     date = st.session_state['date']
     time = st.session_state['time']
     origin = st.session_state['gps_full_pick']
     dest = st.session_state['gps_full_drop']
     trip_distance = st.session_state['tripdistance']


     columns = ['id','date','time','gps_pickup','gps_dropoff',

              'tripdistance']
     
     data = [[id,date,time,origin,dest,trip_distance]]

     #create dataframe
     df = pd.DataFrame(data,columns=columns)

     
     df['date'] = pd.to_datetime(df['date'])
     w_df['date'] = pd.to_datetime(w_df['date'])
     merged_df = pd.merge(df, w_df, on='date', how='left')

     mer_df = cleaner(merged_df)
     
     #Make prediction
     pred = pipeline.predict(mer_df)
     prediction = np.expm1(pred)
     
     df['Pred_ETA']= prediction
     df[['id','tripdistance','Pred_ETA']].to_csv('.\\data\\history.csv',mode='a',header = not os.path.exists('.\\data\\history.csv'),index=False)
     final = df[['id','tripdistance','Pred_ETA']]
     st.dataframe(final)
     return st.session_state['prediction']

# Create a list of hours
hours = [f"{i:02d}:00" for i in range(24)]



#Display form on the streamlit app to take user
def display_form():
     st.title("Make a Prediction")
    

     tab2,tab3 = st.tabs(["### Ride Information","Help"])

     with tab2:
               st.title("Make a Prediction")
               pipeline = select_model()
               st.write ('### Ride Information')
               st.text_input("Ride ID", "Ride order ID",key='id')
               st.date_input("What day was the pickup",value=None, key = 'date')
               st.time_input("What time was the pickup", value=None,key='time')
               st.text_input('Enter your Pickup GPS coordinates', key='gps_full_pick')
               st.text_input('Enter your Dropoff GPS coordinates', key='gps_full_drop')
               st.number_input('Enter trip distance in metres', key='tripdistance', min_value=100, max_value=200000, step=1)

               if st.button('Yassir Predict 👍'):
                    make_prediction(pipeline)
     with tab3:
               st.write ('### Information Security')
               st.write("Please know that information contained in this app has no record of any Yassir client information and we uphold upmost privacy for all users")
               
               
          


with open('frontend/config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)



if __name__ == '__main__':
     
   authenticator = stauth.Authenticate(
   config['credentials'],
   config['cookie']['name'],
   config['cookie']['key'],
   config['cookie']['expiry_days'],
   config['pre-authorized']
   )


authenticator.login(location = 'sidebar')


with st.sidebar:
     uploaded_file = st.file_uploader(
    "Choose a CSV file", accept_multiple_files=False,key='file',type= 'csv'
)

if st.session_state["authentication_status"]:
   authenticator.logout(location = 'sidebar')
   st.write(f'Welcome *{st.session_state["name"]}*')
   

   if uploaded_file is None:
        display_form()
        #st.write(st.session_state['prediction'])
     

   else:
        large_prediction()
     

    
elif st.session_state["authentication_status"] is False:
     st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
     st.warning('Please enter your username and password')



# Add a selectbox to the sidebar:
add_selectbox = st.sidebar.selectbox(
    'How I can be contacted?',
    ('chrappahkwasi@gmail.com','chrappahkwasi@gmail.com', '0209100603')
)


