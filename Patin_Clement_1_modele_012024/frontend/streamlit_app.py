import streamlit as st
import requests

st.title("Air Paradis - twitter sentiment analysis")

col1, col2, col3 = st. columns(3, gap="large")
baseInputs = ["I hate my job", "I love donuts", ""]

for i,(col,baseInput) in enumerate(zip([col1, col2, col3], baseInputs)) :
    with col :

        tweet = st.text_input(label="Tweet :", value=baseInput, key=i)

        st.write("")
        st.write("")


        API_URL = "http://backend:8000"

        response = requests.post(API_URL+"/predict", json={"text":tweet}).json()
        # response = requests.post(API_URL+"/predict?text="+tweet)

        if "score" not in response.keys() :
            st.write(":exclamation: Error :exclamation: :")
            st.write("Detail : "+response["detail"])

        else :
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
