import os
from langchain_ollama import OllamaEmbeddings,ChatOllama
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain



embeddings = OllamaEmbeddings(model="llama3.2")
LLM1 = ChatOllama(model="llama3.2")

document = TextLoader("product-data.txt").load()
text_spliter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_spliter.split_documents(document)
vector_store=Chroma.from_documents(chunks,embeddings)

prompt_template = ChatPromptTemplate.from_messages([
    ("system","""You are an assistant for answering questions. 
    Use the provided context to respond.If the answer  
    isn't clear, acknowledge that you don't know.  
    Limit your response to three concise sentences. 
    {context}
    """),
    ("human","{input}")
])
retriever = vector_store.as_retriever()

qa_chain = create_stuff_documents_chain(LLM1,prompt_template)
rag_chain = create_retrieval_chain(retriever,qa_chain)

print("Chat with documents")
question = input("Your Question: ")

if question:
    response = rag_chain.invoke({"input":question})
    print(response['answer'])
