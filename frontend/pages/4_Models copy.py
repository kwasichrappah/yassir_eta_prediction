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
    pipeline = joblib.load('./models/Xgboost.joblib')
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

        #encoder to inverse transform the result
        #encoder = joblib.load('./models/encoder.joblib')
        return pipeline

def cleaner(df):
    
    df['date'] = pd.to_datetime(df['date'])
    w_df['date'] = pd.to_datetime(w_df['date'])
    merged_df = pd.merge(df, w_df, on='date', how='left')
    merged_df['hour']= merged_df['time'].dt.hour

    #split column origin column into two columns
#     df[['Origin_lat', 'Origin_lon']] = df['origin'].str.split(',', 1, expand=True)
#     df[['Destiation_lat', 'Destination_lon']] = df['dest'].str.split(',', 1, expand=True)
    #Apply cube root transformations
    merged_df['log_Trip_distance'] = np.cbrt(merged_df['Trip_distance'])

    #Smoothening some columns

    # Apply Gaussian filter with a sigma of 1
    merged_df['gaussian_surface_pressure'] = gaussian_filter1d(merged_df['surface_pressure'], sigma=1)
    merged_df['gaussian_mean_sea_level'] = gaussian_filter1d(merged_df['mean_sea_level_pressure'], sigma=1)
    merged_df['gaussian_dewpoint_2m_temperature'] = gaussian_filter1d(merged_df['dewpoint_2m_temperature'], sigma=1)

    merged_df.drop(columns=['Origin_lat','Origin_lon','Destination_lat','Destination_lon','id','date','Trip_distance',
                              'maximum_2m_air_temperature','mean_2m_air_temperature','mean_sea_level_pressure','dewpoint_2m_temperature',
                              'minimum_2m_air_temperature','surface_pressure','u_component_of_wind_10m','v_component_of_wind_10m',
                              'total_precipitation'],axis=1,inplace=True)
    return merged_df

#Prediction and probability variables state at the start of the webapp
if 'prediction' not in st.session_state:
     st.session_state['prediction'] = None



#Making prediction 
def make_prediction(pipeline):
     id = st.session_state['id']
     date= st.session_state['date']
     time = st.session_state['time']
     origin = st.session_state['gps_full_pick']
     dest = st.session_state['gps_full_drop']
     trip_distance = st.session_state['tripdistance']


     columns = ['id','date','time','gps_pickup','gps_dropoff',

              'tripdistance']
     
     data = [[id,date,time,origin,dest,trip_distance]]

     #create dataframe
     df = pd.DataFrame(data,columns=columns)

     # w_df = pd.read_csv('data/Weather_data.csv')
     # df['date'] = pd.to_datetime(df['date'])
     # w_df['date'] = pd.to_datetime(w_df['date'])
     # merged_df = pd.merge(df, w_df, on='date', how='left')

     merged_df = cleaner(data)
     
     



     #df.to_csv('.\\data\\history.csv',mode='a',header = not os.path.exists('.\\data\\history.csv'),index=False)

     #Make prediction
     
     # pred = pipeline.predict(merged_df)
     # prediction = np.expm1(pred)


     #Updating state
     
     #st.session_state['prediction']=prediction
     
     
     

     return df #st.session_state['prediction']

#Display form on the streamlit app to take user
def display_form():
     pipeline = select_model()

     with st.form('input-features'):
          col1,col2 = st.columns(2)

          with col1:
               st.write ('### Ride Information')
               st.text_input("Ride ID", "Ride order ID",key='id')
               st.date_input("What day was the pickup", value=None, key = 'date')
               st.time_input("What time was the pickup", value=None,key='time')
               st.text_input('Enter your Pickup GPS coordinates', key='gps_full_pick')
               st.text_input('Enter your Dropoff GPS coordinates', key='gps_full_drop')
               st.number_input('Enter trip distance in metres', key='tripdistance', min_value=100, max_value=200000, step=1)
               


          st.form_submit_button('Predict',on_click = make_prediction,kwargs = dict(pipeline = pipeline))



with open('frontend/config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)



if __name__ == '__main__':
     
#    authenticator = stauth.Authenticate(
#    config['credentials'],
#    config['cookie']['name'],
#    config['cookie']['key'],
#    config['cookie']['expiry_days'],
#    config['pre-authorized']
#    )


# authenticator.login(location='sidebar')

# if st.session_state["authentication_status"]:
#    authenticator.logout(location = 'sidebar')
#   st.write(f'Welcome *{st.session_state["name"]}*')
   st.title("Make a Prediction")
   m=display_form()

   st.write(m)#(st.session_state['prediction'])



    
# elif st.session_state["authentication_status"] is False:
#     st.error('Username/password is incorrect')
# elif st.session_state["authentication_status"] is None:
#     st.warning('Please enter your username and password')


# Add a selectbox to the sidebar:
add_selectbox = st.sidebar.selectbox(
    'How I can be contacted?',
    ('chrappahkwasi@gmail.com','chrappahkwasi@gmail.com', '0209100603')
)