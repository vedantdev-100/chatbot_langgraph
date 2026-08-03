import streamlit as st

with st.chat_message('user'):
    st.text("hi")
    
with st.chat_message('assistant'):
    st.text("how can I help you?")
    
user_input = st.chat_input("Type here") 

if user_input:
    with st.chat_message('user'):
        st.text(user_input)
