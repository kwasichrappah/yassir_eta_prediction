import streamlit as st
import pandas as pd
import matplotlib as plt 
import joblib
import sys
import numpy as np
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader


# Streamlit page configuration
st.set_page_config(page_title="Data", page_icon="💾", layout="wide")



with open('frontend/config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

if __name__ == "__main__":

    
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
    
    st.header("Ride user information")

    st.write("This is data is not meant for the public since it contains some private information about people.")


    st.caption('Data was gathered from :red[Yassir Ride Hailing Company]')


        
elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
    st.session_state.clear()
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')
    st.session_state.clear()

# Add a selectbox to the sidebar:
add_selectbox = st.sidebar.selectbox(
    'How I can be contacted?',
    ('chrappahkwasi@gmail.com','chrappahkwasi@gmail.com', '0209100603')
)