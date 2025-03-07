import os
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

embedding_model = OllamaEmbeddings(model="llama3.2")

document = TextLoader("job_listings.txt").load()
text_spliter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
chunks = text_spliter.split_documents(document)
db=FAISS.from_documents(chunks,embedding_model)

retriever = db.as_retriever()

text = input("Enter the query")
# ---------------------------------------------------
docs = retriever.invoke(text)
# --------------------------------------------------
# or
# -------------------------------------------------
# embedded_vector = embedding_model.embed_query(text)

# docs = db.similarity_search_by_vector(embedded_vector)
# -----------------------------------------------------
for doc in docs:
    print(doc.page_content)