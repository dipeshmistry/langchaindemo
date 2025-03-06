import os

import inputs
from langchain_ollama import ChatOllama
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.schema.runnable import RunnableLambda
from mistralai import Mistral
from langchain.globals import set_debug

from simple_sequential_chain_demo import speach_template

set_debug(True)
# openai_api_key = os.  getenv("OPENAI_API_KEY")
# LLM1 = ChatOllama(model="llama3.2")

api_key = os.getenv("MISTRAL_API_KEY")
model = "mistral-large-latest"
LLM1 = ChatOllama(model="llama3.2")
mistral_client = Mistral(api_key=api_key)

st.title("Speach Generator")

subject_template = PromptTemplate(
    input_variables=["product_name", "features"],
    template="""You are an experienced marketing specialist.  
    Create a catchy subject line for a marketing  
    email promoting the following product: {product_name}.  
    Highlight these features: {features}.  
    Respond with only the subject line."""
)

email_template = PromptTemplate(
    input_variables=["product_name", "subject_line", "target_audience"],
    template="""Write a marketing email of 300 words for the  
    product: {product_name}. Use the subject line: 
     {subject_line}. Tailor the message for the  
     following target audience: {target_audience}. 
      Format the output as a JSON object with three  
      keys: 'subject', 'audience', 'email' and fill  
      them with respective values."""
)


def generate_speech(input):
    email_prompt = email_template.format(product_name=product_name,subject_line=input,target_audience=target_audience)

    email_response = mistral_client.chat.complete(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": email_prompt}],
    )
    return email_response.choices[0].message.content


# simple sequential chain
first_chain = subject_template | LLM1 | StrOutputParser() | (
    lambda subject_line: (st.write(subject_line), subject_line)[1])
second_chain = RunnableLambda(generate_speech) | JsonOutputParser()
final_chain = first_chain | second_chain

product_name = st.text_input("Enter the product_name: ")
features = st.text_input("Enter the features: ")
target_audience = st.text_input("Enter the target_audience: ")

if product_name and features:
    response = final_chain.invoke({"product_name": product_name, "features": features})
    st.write(response)
