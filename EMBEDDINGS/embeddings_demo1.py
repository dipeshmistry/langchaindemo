import os
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI

# openai_api_key = os.getenv("OPENAI_API_KEY")
LLM = ChatOllama(model="gemma:2b")
# LLM = ChatOpenAI(model="gemma:2b")

question = input("Enter the question")
response = LLM.invoke(question)
print(response.content)