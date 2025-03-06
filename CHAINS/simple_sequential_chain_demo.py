import os

import streamlit
from langchain_ollama import ChatOllama
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# openai_api_key = os.  getenv("OPENAI_API_KEY")
LLM = ChatOllama(model="llama3.2")

st.title("Speach Generator")

title_template = PromptTemplate(
    input_variables=["topic"],
    template="""You are an experienced speech writer. 
    You need to craft an impactful title for a speech  
    on the following topic: {topic} 
    Answer exactly with one title.
    """
)

speach_template = PromptTemplate(
    input_variables=["title"],
    template="""You need to write a powerful speech of 350 words 
     for the following title: {title}
    """
)

# simple sequential chain
first_chain = title_template | LLM | StrOutputParser() | (lambda title: (st.write(title),title)[1])
second_chain = speach_template | LLM
final_chain = first_chain | second_chain

topic = st.text_input("Enter the topic: ")

if topic:
    response = final_chain.invoke({"topic": topic})
    st.write(response.content)
