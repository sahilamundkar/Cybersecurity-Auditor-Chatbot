import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader



#--USER AUTHENTICATION--
with open(r'C:\Users\hp\Project\Recipe\Cybersecurity project\config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['pre-authorized']
)

authenticator.login("sidebar")

if st.session_state["authentication_status"]:
    authenticator.logout( 'Logout', 'sidebar')
    st.write(f'Welcome *{st.session_state["name"]}*')
    
elif st.session_state["authentication_status"] is False:
    with st.sidebar:
        st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    with st.sidebar:
        st.warning('Please enter your username and password')



footer = """
<style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: white;
        color: black;
        text-align: center;
    }
</style>
<div class="footer">
    <p>Aegis is based on Llama3 LLM, however it can make mistake. Please verify ISO Document for Important Information</p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
