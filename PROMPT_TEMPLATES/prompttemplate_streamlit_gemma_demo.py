import os

from langchain.chains.summarize.map_reduce_prompt import prompt_template
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain.globals import set_debug
import streamlit as st
from langchain.prompts import PromptTemplate


set_debug(False)

st.title('Cuisine Info')

country = st.text_input("Enter the country: ")
no_of_paras = st.number_input("Enter no of paras: ",min_value=1,max_value=5)
lang = st.text_input("Enter the language: ")
# openai_api_key = os.getenv("OPENAI_API_KEY")
# LLM = ChatOllama(model="gemma:2b")
LLM = ChatOllama(model="gemma:2b", system="Answer strictly in "+ lang +". Do not use English.")
# LLM = ChatOpenAI(model="gemma:2b")

prompt_template = PromptTemplate(
    input_variables=["country","no_of_paras","language"],
    template="""You are and expert in traditional cuisines.
    You provide imformation about the specific dish from a specific country
    Answer the question: What is the traditional cuisine of {country}?
    Answer in {no_of_paras} short paras in {language}.
    Your answer **must** be in {language}. If language is Hindi, use Devanagari script.
    """
)
# prompt_template = PromptTemplate(
#     input_variables=["country","no_of_paras","language"],
#     template="""You are an expert in traditional cuisines.
#     Provide information about traditional dishes from {country}.
#     Your answer **must** be in {language}. If language is Hindi, use Devanagari script.
#     """
#
# )


print(prompt_template.format(country=country, no_of_paras = no_of_paras, language=lang))
response = LLM.invoke(prompt_template.format(country=country, no_of_paras = no_of_paras, language=lang))
st.write(response.content)
print(response.content)