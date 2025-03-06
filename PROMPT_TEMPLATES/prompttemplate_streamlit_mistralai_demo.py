

import os
from mistralai import Mistral
import streamlit as st
from langchain.prompts import PromptTemplate

api_key = os.getenv("MISTRAL_API_KEY")
# print(api_key)
model = "mistral-large-latest"

st.title("Cuisine Info ")

prompttemplate = PromptTemplate(
    input_variables=["country", "no_of_paras", "language"],
    template="""You are and expert in traditional cuisines.
        You provide imformation about the specific dish from a specific country
        Avoid giving answers to the frictional places, if the country is frictional or non-existant answer: I don't know.
        Answer the question: What is the traditional cuisine of {country}?
        Answer in {no_of_paras} short paras in {language} only if country is not frictional.
        Your answer **must** be in {language}. If language is Hindi, use Devanagari script.
        """
)

country = st.text_input("Enter the country: ")
no_of_paras = st.number_input("Enter no of paras: ",min_value=1,max_value=5)
lang = st.text_input("Enter the language: ")

client = Mistral(api_key=api_key)
# question = st.text_input("Type Your Question :   ")
chat_response = client.chat.complete(
    model= model,
    messages = [
        {
            "role": "user",
            "content": prompttemplate.format(country=country, no_of_paras = no_of_paras, language=lang),
        },
    ]
)

st.write(chat_response.choices[0].message.content)
print(chat_response.choices[0].message.content)