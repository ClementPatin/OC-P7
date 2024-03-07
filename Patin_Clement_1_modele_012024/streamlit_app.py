import streamlit as st
import requests

st.title("Air Paradis - twitter sentiment analysis")

col1, col2, col3 = st. columns(3)
baseInputs = ["I hate my job", "I love donuts", "my shirt is yellow"]

for i,(col,baseInput) in enumerate(zip([col1, col2, col3], baseInputs)) :
    with col :

        tweet = st.text_input(label="Tweet :", value=baseInput, key=i)

        st.write("Your tweet submission is :", tweet)


        API_URL = "http://localhost:8000"

        response = requests.post(API_URL+"/predict", json={"text":tweet})
        # response = requests.post(API_URL+"/predict?text="+tweet)

        st.write(response.json())



# streamlit run streamlit_app.py
