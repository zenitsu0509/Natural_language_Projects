import streamlit as st
from mylib import predict
import streamlit.components.v1 as components


st.set_page_config(
    page_title="Sentiment Analysis", 
    page_icon=":sunglasses:"
    )

st.title("😎 Movie Review Rating Predictor")

review = st.text_area("Write your movie review below, and we'll predict the rating for you!")

if st.button("Predict"):
    result = predict({'review': review})
    st.write("### The predicted rating is" )
    st.write(f"# {result['rating']}")
