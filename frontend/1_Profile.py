import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader



# Set page config
st.set_page_config(
    page_title="Profile",
    page_icon="🛖",
    layout="wide",
)


with open('./frontend/config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)


authenticator = stauth.Authenticate(
config['credentials'],
config['cookie']['name'],
config['cookie']['key'],
config['cookie']['expiry_days'],
config['pre-authorized']
)


authenticator.login(location='sidebar')

if st.session_state["authentication_status"]:
    authenticator.logout()
    st.write(f'Welcome *{st.session_state["name"]}*')
    st.title ("Yassir ETA Predictor WebApp")
    st.write("This is a ML model application that predicts exact time of arrival for Yassir customers.")
    st.write("It uses ML algorithms to make these predictions.")
    tab1, tab2, tab3 = st.tabs(["Problem Statement","Key Features", "Key Metrics and Success Criteria"])

    with tab1:
        st.subheader('Problem Statement', divider='red')
        st.subheader("This project aims to :rainbow[accurately predict ETA] to improve reliability and efficiency for Yassir Company in this competitve ride hailing market .")
        st.markdown("In the ride haling market, customers are bundled with many ride hailing apps with different selling points ranging from pricing,car comfort,\
                    discount offers to early arrival. Yassir wants to stand out with being able to accurately predict ETA before being able to offer discounts.")
        
    with tab2:
       st.subheader('Key Features in the model', divider='orange')
       st.markdown("- Ride Order ID")
       st.markdown("- Time of Pickup")
       st.markdown("- Location of Pickup")
       st.markdown("- Location of Dropoff ")
       st.markdown("- Ride Duration")

    with tab3:
       st.subheader("Key Metrics and Success Criteria",divider = "green")
       st.markdown("• Model Accuracy : Model Accuracy is based on the rmse.")
       st.markdown("• Model Interpretability : The degree to which the model’s predictions and insights can be understood and utilized by stakeholders.")
       

    
    # Inject custom CSS for styling
    st.markdown(
        """
        <style>
        .custom-text {
            font-size: 12px; /* Adjust the font size as needed */
            text-align: right;
            color: #000; /* Optionally, you can also change the text color */
            margin: 0; /* Adjust margin as needed */
            padding: 0; /* Adjust padding as needed */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Create text input and apply custom class
    user_input = 'Created by Emmanuel Chrappah'

    # Apply the custom class to the displayed text
    st.markdown(f'<p class="custom-text">{user_input}</p>', unsafe_allow_html=True)

elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')








# Add a selectbox to the sidebar:
add_selectbox = st.sidebar.selectbox(
    'How I can be contacted?',
    ('chrappahkwasi@gmail.com','chrappahkwasi@gmail.com', '0209100603')
)
        





















#streamlit run "c:/Users/chrap/OneDrive - ECG Ghana/Emmanuel Chrappah/Azubi Africa/git_hub_repos/streamlit_fundamentals/src/app.py"     