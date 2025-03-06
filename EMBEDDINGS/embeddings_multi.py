import os
from langchain_ollama import OllamaEmbeddings

embedding_model = OllamaEmbeddings(model="llama3.2")

text = input("Enter the text")
response = embedding_model.embed_query(text)
print(response)