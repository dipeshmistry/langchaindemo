import os
from langchain_ollama import OllamaEmbeddings

embedding_model = OllamaEmbeddings(model="llama3.2")

text = input("Enter the text (comma-separated): ")
text_array = text.split(',')
text_array = [item.strip() for item in text_array]


response = embedding_model.embed_documents(text_array)
print(response)