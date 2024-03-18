import streamlit as st
import requests

st.title("Air Paradis - twitter sentiment analysis_test2")

col1, col2, col3 = st. columns(3, gap="large")
baseInputs = ["I hate my job", "I love donuts", "my shirt is yellow"]

for i,(col,baseInput) in enumerate(zip([col1, col2, col3], baseInputs)) :
    with col :

        tweet = st.text_input(label="Tweet :", value=baseInput, key=i)

        st.write("")
        st.write("")


        API_URL = "https://testapip7.azurewebsites.net"

        response = requests.post(API_URL+"/predict", json={"text":tweet}).json()
        # response = requests.post(API_URL+"/predict?text="+tweet)

        st.write("Score :")
        st.progress(float(response["score"]), text = response["score"])
        st.write("")
        st.write("")

        st.write("Sentiment :")
        if response["sentiment"] == "positive" :
            st.header(":smile:")
        else :
            st.header("	:angry:")



# streamlit run streamlit_app.py
            

# 
