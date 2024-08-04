import streamlit as st
import time
import plotly.express as px
import pandas as pd
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

# Set page config
st.set_page_config(
    page_title="Dashboard",
    page_icon="📉",
    layout="wide",
)


def loader():

  # Add a placeholder
  bar = st.progress(0)

  for i in range(100):
    # Update the progress bar with each iteration.
    bar.progress(i + 1)
    time.sleep(0.1)

df= pd.read_csv("./data/train_data.csv")


def eda_dashboard():
   st.markdown('### Exploratory Data Analysis Dashboard')
   loader()
   col1,col3 = st.columns(2)

   with col1:
      int_service_histogram = px.histogram(df,x='Tripdistance',color = 'Churn')

      st.plotly_chart(int_service_histogram)

   with col3:
       contract_histogram = px.histogram(df,x='ETA',color = 'Churn')

       st.plotly_chart(contract_histogram)  

   






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
   st.title("Dashboard")

   col1,col2 = st.columns(2)
   with col1:
      pass
   with col2:
      st.selectbox('Select the type of Dashboard',options=['EDA'],key='dashboard_type')

   if st.session_state['dashboard_type'] == 'EDA':
    eda_dashboard()

   else:
    pass

    
elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')








# Add a selectbox to the sidebar:
add_selectbox = st.sidebar.selectbox(
    'How I can be contacted?',
    ('chrappahkwasi@gmail.com','chrappahkwasi@gmail.com', '0209100603')
)
        


