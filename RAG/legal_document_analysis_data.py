import os
import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_ollama import OllamaEmbeddings,ChatOllama
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain,create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.globals import set_debug

set_debug(True)

# openai_api_key = os.getenv("OPENAI_API_KEY")
# embeddings = OpenAIEmbeddings(api_key=openai_api_key)
# LLM1 = ChatOpenAI(model="gpt-4o" ,api_key=openai_api_key)

embeddings = OllamaEmbeddings(model="llama3.2")
LLM1 = ChatOllama(model="llama3.2")

document = TextLoader("legal_document_analysis_data.txt").load()
text_spliter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_spliter.split_documents(document)
vector_store=Chroma.from_documents(chunks,embeddings)

prompt_template = ChatPromptTemplate.from_messages([
    ("system","""You are an assistant for answering questions. 
    Use the provided context to respond.If the answer  
    isn't clear, acknowledge that you don't know.  
    Limit your response to three concise sentences. 
    IMPORTANT: Only answer using the context provided. If the answer isn’t clearly in the context, say: 'I apologize, but that falls outside of my current scope of knowledge.'

    {context}
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human","{input}")
])
retriever = vector_store.as_retriever()
st.write(retriever)
history_aware_retriever = create_history_aware_retriever(LLM1,retriever,prompt_template)
qa_chain = create_stuff_documents_chain(LLM1,prompt_template)
rag_chain = create_retrieval_chain(history_aware_retriever,qa_chain)

history_for_chain = StreamlitChatMessageHistory()
chain_with_history = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id:history_for_chain,
    input_messages_key="input",
    history_messages_key="chat_history"
)

st.write("Chat with documents")
question = st.text_input("Your Question: ")

if question:
    response = chain_with_history.invoke({"input":question},{"configurable":{"session_id":"abc123"}})
    st.write(response['answer'])
