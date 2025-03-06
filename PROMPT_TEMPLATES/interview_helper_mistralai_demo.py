import os

from mistralai import Mistral
import streamlit as st
from langchain.prompts import PromptTemplate

api_key = os.getenv("MISTRAL_API_KEY")
# print(api_key)
model = "mistral-large-latest"

st.title("Interview Helper.")

prompttemplate = PromptTemplate(
    input_variables=["position", "company", "strengths", "weaknesses"],
    template="""You are a career coach. Provide tailored interview tips for the 
    position of {position} at {company}. 
    Highlight your strengths in {strengths} and prepare for questions 
    about your weaknesses such as {weaknesses}.
        """
)

position = st.text_input("Enter your position: ")
company = st.text_input("Enter company name: ")
strengths = st.text_input("What are your strengths: ")
weaknesses = st.text_input("What are your weaknesses: ")

client = Mistral(api_key=api_key)
# question = st.text_input("Type Your Question :   ")
if position and company and strengths and weaknesses:
    chat_response = client.chat.complete(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompttemplate.format(position=position, company=company, strengths=strengths, weaknesses=weaknesses),
            },
        ]
    )

    st.write(chat_response.choices[0].message.content)
    print(chat_response.choices[0].message.content)
