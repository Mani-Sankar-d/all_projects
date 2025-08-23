import streamlit as st,requests

st.title("Youtube transcript Chatbot")

id = st.text_input("Please enter the video id")
query = st.text_input("Query")

url = "http://127.0.0.1:8000"

if "a" not in  st.session_state:
    st.session_state.a = st.button("Generate")
else:
    parameters = {"query":query,"id":id}
    response = requests.post(f"{url}/chat",json=parameters)
    st.text_area("Output", response.json()["response"])
    st.button("Done")
