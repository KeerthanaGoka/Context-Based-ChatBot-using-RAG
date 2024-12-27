import streamlit as st
import re
import uuid

userinfo    = {"kmlhan@clarku.edu":        "kunal", 
               "kgka@clarku.edu":          "keerthana"}

def main():
    project_title       = 'Kanji Chatbot'
    st.header(project_title)

    loginPageContent    = st.empty()
    with loginPageContent:
        st.write("Login Information")
        with st.form(key='login_form'):

            userid          = st.text_input("User ID (Clark Email address):")
            password        = st.text_input("Password (Your firstname):", type="password")

            submitResponse  = st.form_submit_button(label="Login", type="primary", use_container_width=True)
            if submitResponse:
                if userid in userinfo and userinfo[userid] == password:
                    st.success("Login successful! Welcome, {}".format(userid))
                    st.session_state.page           = 1
                    st.session_state['sessionID']   = re.sub(r'[^a-zA-Z0-9]', '_', userid).lower()
                    st.session_state['userID']      = userid
                    loginPageContent.empty()
                    st.rerun()
                else:
                    st.error("Invalid username or password. Please try again.")
                    
            defaultResponse = st.form_submit_button(label="Continue without login", type="secondary", use_container_width=True)
            if defaultResponse:
                st.session_state.page               = 1
                st.session_state['sessionID']       = 'default_' + (uuid.uuid4().hex).lower()
                st.session_state['userID']          = 'Anonymous'
                loginPageContent.empty()
                st.rerun()
