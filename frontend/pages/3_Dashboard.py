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

df= pd.read_csv("./data/train_data.csv",parse_dates = ['Timestamp'])
df['date'] = df['Timestamp'].dt.date
df['hour'] = df['Timestamp'].dt.hour
df= df.sort_values(by='Timestamp')


def eda_dashboard():
   st.markdown('### Exploratory Data Analysis Dashboard')
   col1,col3 = st.columns(2)

   with col1:
      data = df.head(50000)
      trip_distance_histogram = px.bar(data,x='date',y='Trip_distance',hover_data=['ETA'],color='ETA',title="Daily total trip distances")

      st.plotly_chart(trip_distance_histogram)

   with col3:
       eta_barchart = px.bar(data,x='date',y='ETA',hover_data =['Trip_distance'],color='Trip_distance',title="Daily sum of ETA")

       st.plotly_chart(eta_barchart)  

   eta_distance_barchart = px.bar(data,x='ETA',y='Trip_distance',hover_data =['hour'],color ="hour",title="ETA against Trip Distance")

   st.plotly_chart(eta_distance_barchart) 


def pickups():
   
   pickups = df[['Origin_lat','Origin_lon']].head(1000)
   pickups.rename(columns={"Origin_lat": "lon", "Origin_lon": "lat"},inplace=True)
   st.map(pickups,color= '#00ff00',size = 10)

def dropoffs():
   dropoffs = df[['Destination_lat','Destination_lon']].head(1000)
   dropoffs.rename(columns={"Destination_lat": "lon", "Destination_lon": "lat"},inplace=True) 
   st.map(dropoffs,color= '#ff0000',size = 10)



# with open('frontend/config.yaml') as file:
#     config = yaml.load(file, Loader=SafeLoader)


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
#    st.write(f'Welcome *{st.session_state["name"]}*')
    st.title("Dashboard")

    col1,col2 = st.columns(2)
    with col1:
        pass
    with col2:
        st.selectbox('Select the type of Dashboard',options=['EDA','LOCATIONS'],key='dashboard_type')

    if st.session_state['dashboard_type'] == 'EDA':
        eda_dashboard()

    else:
        col1,col2 = st.tabs(["Pick up points", "Dropoff Points"])

        with col1:
            st.write ('### Information on some Pickup points')
            pickups()
            
            
            
        with col2:
            st.write('### Information on some Dropoff points')
            dropoffs()
            

    
# elif st.session_state["authentication_status"] is False:
#     st.error('Username/password is incorrect')
# elif st.session_state["authentication_status"] is None:
#     st.warning('Please enter your username and password')








# # Add a selectbox to the sidebar:
# add_selectbox = st.sidebar.selectbox(
#     'How I can be contacted?',
#     ('chrappahkwasi@gmail.com','chrappahkwasi@gmail.com', '0209100603')
# )
        


