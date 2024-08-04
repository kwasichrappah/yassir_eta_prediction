import streamlit as st
import pandas as pd
import joblib
import os
import datetime
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

#setting page title and icon
st.set_page_config(
    page_title = "Prediction Page",
    page_icon = "🎯",
    layout = 'wide'
)

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
        return pipeline,encoder


#Prediction and probability variables state at the start of the webapp
if 'prediction' not in st.session_state:
     st.session_state['prediction'] = None



#Making prediction 
def make_prediction(pipeline):
     ride_id = st.session_state['SeniorCitizen']
      = st.session_state['Partner']
     dependents = st.session_state['Dependents']
     phoneservice = st.session_state['PhoneService']
     multiplelines = st.session_state['MultipleLines']
     InternetService = st.session_state['InternetService']
     onlinesecurity = st.session_state['OnlineSecurity']
     onlinebackup = st.session_state['OnlineBackup']
     deviceprotetion = st.session_state['DeviceProtection']
     techsupport = st.session_state['TechSupport']
     streamingtv = st.session_state['StreamingTV']
     streamingmovies = st.session_state['StreamingMovies']
     contract = st.session_state['Contract']
     paperlessbilling = st.session_state['PaperlessBilling']
     tenure = st.session_state['tenure']
     monthlycharges = st.session_state['MonthlyCharges']
     paymentmethod = st.session_state['PaymentMethod']

     columns = ['SeniorCitizen','Partner','Dependents','PhoneService','MultipleLines',

              'InternetService','OnlineSecurity','OnlineBackup','DeviceProtection',

              'TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod','MonthlyCharges','tenure']
     
     data = [[SeniorCitizen,partner,dependents,phoneservice,multiplelines,
              InternetService,onlinesecurity,onlinebackup,deviceprotetion,
              techsupport,streamingtv,streamingmovies,contract,paperlessbilling,paymentmethod,monthlycharges,tenure]]
     
     #create dataframe
     df = pd.DataFrame(data,columns=columns)



     df.to_csv('.\\data\\history.csv',mode='a',header = not os.path.exists('.\\data\\history.csv'),index=False)

     #Make prediction
     
     pred = pipeline.predict(df)
     prediction = int(pred[0])


     #Updating state
     if  prediction == 1:
        st.session_state['prediction']='Yes'
     else:
          st.session_state['prediction'] ='No'
     

     return st.session_state['prediction']

#Display form on the streamlit app to take user
def display_form():
     pipeline = select_model()

     with st.form('input-features'):
          col1,col2 = st.columns(2)

          with col1:
               st.write ('### Ride Information')
               st.selectbox('Senior Citizen',['Yes','No'],key='SeniorCitizen')
               st.selectbox('Gender',['Male','Female'],key='gender')
               st.selectbox('Dependents',['Yes','No'],key='Dependents')
               st.selectbox('Partner',['Yes','No'],key='Partner')
               st.selectbox('Phone Service',['Yes','No'],key='PhoneService')
               st.selectbox('Multiple Lines',['Yes','No'],key='MultipleLines')
               st.selectbox('Internet Service',['Fiber Optic','DSL'],key='InternetService')


          with col2:
               st.write('### Work Information')
               st.selectbox('Online Security',['Yes','No'],key='OnlineSecurity')
               st.selectbox('Online Backup',['Yes','No'],key='OnlineBackup')
               st.selectbox('Device Protection',['Yes','No'],key='DeviceProtection')
               st.selectbox('Tech Support',['Yes','No'],key='TechSupport')
               st.selectbox('Streaming TV',['Yes','No'],key='StreamingTV')
               st.selectbox('Streaming Movies',['Yes','No'],key='StreamingMovies')
               st.selectbox('Contract Type',['Month-to-month','One year','Two year'],key='Contract')
               st.selectbox('Paperless Billing',['Yes','No'],key='PaperlessBilling')
               st.selectbox('What is your payment method', options=['Electronic Check','Mailed check', 'Bank transfer', 'Credit Card']
                            ,key='PaymentMethod')
               st.number_input('Enter your monthly charge', key='MonthlyCharges', min_value=10, max_value=200, step=1)
               st.number_input('Enter Tenure in months', key = 'tenure', min_value=2, max_value=72, step=1)
               


          st.form_submit_button('Predict',on_click = make_prediction,kwargs = dict(pipeline = pipeline,encoder=encoder))



with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)



if __name__ == '__main__':
     
   authenticator = stauth.Authenticate(
   config['credentials'],
   config['cookie']['name'],
   config['cookie']['key'],
   config['cookie']['expiry_days'],
   config['pre-authorized']
   )


authenticator.login(location='sidebar')

if st.session_state["authentication_status"]:
   authenticator.logout(location = 'sidebar')
   st.write(f'Welcome *{st.session_state["name"]}*')
   st.title("Make a Prediction")
   display_form()

   st.write(st.session_state['prediction'])



    
elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')


# Add a selectbox to the sidebar:
add_selectbox = st.sidebar.selectbox(
    'How I can be contacted?',
    ('chrappahkwasi@gmail.com','chrappahkwasi@gmail.com', '0209100603')
)