import streamlit as st
from chatbot import page_login, page_chatbot

if "page" not in st.session_state:
    st.session_state['sessionID']   = 'default'
    st.session_state.page           = 0

if st.session_state.page == 0:
    st.session_state['sessionFromChatPage']   = False
    page_login.main()
elif st.session_state.page == 1:
    page_chatbot.main()

