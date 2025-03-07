import os
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain.globals import set_debug
import streamlit as st

set_debug(True)

st.title('Ask Anything')

# openai_api_key = os.getenv("OPENAI_API_KEY")
LLM = ChatOllama(model="gemma:2b")
# LLM = ChatOpenAI(model="gemma:2b")

question = st.text_input("Enter the question")
response = LLM.invoke(question)
st.write(response.content)
print(response.content)