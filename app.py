import streamlit as st

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    if st.button("Log in"):
        st.session_state.logged_in = True
        st.rerun()

def logout():
    if st.button("Log out"):
        st.session_state.logged_in = False
        st.rerun()

login_page = st.Page(login, title="Log in", icon=":material/login:")
logout_page = st.Page(logout, title="Log out", icon=":material/logout:")

dashboard = st.Page(
    "./Chatbot2.py", title="Aegis", icon=":material/dashboard:", default=True
)
contact = st.Page("Contact/contact.py", title="Contact us", icon=":material/contacts_product:")
profile = st.Page("authentication/login.py", title="Profile", icon=":material/logout:")
about = st.Page("Contact/about.py", title="About", icon=":material/info_i:")
login_page_nav = st.Page(
    "authentication/login.py", title="Login", icon=":material/notification_important:"
)

# search = st.Page("tools/search.py", title="Search", icon=":material/search:")
# history = st.Page("tools/history.py", title="History", icon=":material/history:")

if st.session_state.logged_in:
    pg = st.navigation(
        {
            
            "Dashboard": [dashboard],
            "Tools": [contact, about],
            "Account": [logout_page]
        }
    )
else:
    pg = st.navigation([login_page])

pg.run()

#--Foter--

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