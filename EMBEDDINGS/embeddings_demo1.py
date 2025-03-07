import os
from langchain_community.chat_models import ChatOllama
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, embeddings, OpenAIEmbeddings

openai_api_key = os.getenv("OPENAI_API_KEY")
api_key = os.getenv("MISTRAL_API_KEY")
embedding_model = OpenAIEmbeddings(api_key=openai_api_key)
# Initialize free embeddings model (from Hugging Face)
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# LLM = ChatOllama(model="gemma:2b")
# LLM = ChatOpenAI(model="gemma:2b")

text = input("Enter the text")
response = embedding_model.embed_query(text)
print(response)