import streamlit as st
from mylib import predict
import streamlit.components.v1 as components


st.set_page_config(
    page_title="Sentiment Analysis", 
    page_icon=":sunglasses:"
    )

st.title("Sentiment Analysis")
st.write("This is a simple example of how to use Streamlit to perform sentiment analysis on text data.")

review = st.text_area("Enter your review here:")

if st.button("Predict"):
    result = predict({'review': review})
    st.write("### The predicted rating is" )
    st.write(f"# {result['rating']}")
