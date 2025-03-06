import os
from mistralai import Mistral
import streamlit as st
from langchain.prompts import PromptTemplate

api_key = os.getenv("MISTRAL_API_KEY")
# print(api_key)
model = "mistral-large-latest"

st.title("Travel Guide: ")

prompttemplate = PromptTemplate(
    input_variables=["city", "month", "language", "budget"],
    template="""Welcome to the {city} travel guide! 
    If you're visiting in {month}, here's what you can do: 
    1. Must-visit attractions. 
    2. Local cuisine you must try. 
    3. Useful phrases in {language}. 
    4. Tips for traveling on a {budget} budget. 
    Enjoy your trip!
        """
)

city = st.text_input("Enter the country: ")
month = st.text_input("Enter the month: ")
lang = st.text_input("Enter the language: ")
budget = st.selectbox("Select budget type: ", ["Low","Medium","High"])

client = Mistral(api_key=api_key)
# question = st.text_input("Type Your Question :   ")
if city and budget and month and lang:
    chat_response = client.chat.complete(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompttemplate.format(city=city, month=month, language=lang, budget=budget),
            },
        ]
    )

    st.write(chat_response.choices[0].message.content)
    print(chat_response.choices[0].message.content)
