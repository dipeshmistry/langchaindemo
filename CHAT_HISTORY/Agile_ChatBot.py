import os

import inputs
from langchain_ollama import ChatOllama
import streamlit as st
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from mistralai import Mistral

# from langchain.globals import set_debug

# set_debug(True)
# openai_api_key = os.  getenv("OPENAI_API_KEY")
# LLM1 = ChatOllama(model="llama3.2")

api_key = os.getenv("MISTRAL_API_KEY")
model = "mistral-large-latest"
LLM1 = ChatOllama(model="llama3.2")
mistral_client = Mistral(api_key=api_key)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a Agile Coach answer any questions realated to the Agile Process"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ]
)

st.title("Agile Guide")

# simple sequential chain
first_chain = prompt_template | LLM1

history_for_chain = StreamlitChatMessageHistory()

chain_with_history = RunnableWithMessageHistory(
    first_chain,
    lambda sessionId: history_for_chain,
    input_messages_key="input",
    history_messages_key="chat_history"
)

#     question =input["Enter the Question:"]
# while True:
input = st.text_input("Enter the Question: ")
if input:
    response = chain_with_history.invoke(
        {"input": input},
        {"configurable": {"session_id": "abc123"}}
    )
    st.write(response.content)

st.write("HISTORY")
st.write(history_for_chain)
