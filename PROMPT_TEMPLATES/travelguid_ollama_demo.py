import os
from langchain_ollama import ChatOllama
import streamlit as st
from langchain.prompts import PromptTemplate

# openai_api_key = os.  getenv("OPENAI_API_KEY")
LLM = ChatOllama(model="llama3.2")

st.title("Cuisine Info")

prompt_template = PromptTemplate(
    input_variables=["country","no_of_paras","language"],
    template="""You are and expert in traditional cuisines.
    You provide imformation about the specific dish from a specific country
    Answer the question: What is the traditional cuisine of {country}?
    Answer in {no_of_paras} short paras in {language}.
    Your answer **must** be in {language}. If language is Hindi, use Devanagari script.
    """
)

country = st.text_input("Enter the country: ")
no_of_paras = st.number_input("Enter no of paras: ",min_value=1,max_value=5)
lang = st.text_input("Enter the language: ")

print(prompt_template.format(country=country, no_of_paras = no_of_paras, language=lang))
response = LLM.invoke(prompt_template.format(country=country, no_of_paras = no_of_paras, language=lang))
st.write(response.content)
print(response.content)