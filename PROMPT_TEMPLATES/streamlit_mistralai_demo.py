

import os
from mistralai import Mistral
import streamlit as st

api_key = os.getenv("MISTRAL_API_KEY")
print(api_key)
model = "mistral-large-latest"

st.title("Ask Anything")

client = Mistral(api_key=api_key)
question = st.text_input("Type Your Question :   ")
chat_response = client.chat.complete(
    model= model,
    messages = [
        {
            "role": "user",
            "content": question,
        },
    ]
)

st.write(chat_response.choices[0].message.content)
print(chat_response.choices[0].message.content)