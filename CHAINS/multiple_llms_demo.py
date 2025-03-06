import os

from langchain_ollama import ChatOllama
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.schema.runnable import RunnableLambda
from mistralai import Mistral
from langchain.globals import set_debug

set_debug(True)
# openai_api_key = os.  getenv("OPENAI_API_KEY")
# LLM1 = ChatOllama(model="llama3.2")

api_key = os.getenv("MISTRAL_API_KEY")
model = "mistral-large-latest"
LLM1 = ChatOllama(model="llama3.2")
mistral_client = Mistral(api_key=api_key)

st.title("Speach Generator")

title_template = PromptTemplate(
    input_variables=["topic"],
    template="""You are an experienced speech writer. 
    You need to craft an impactful title for a speech  
    on the following topic: {topic} 
    Answer exactly with one title.
    """
)

speech_template = PromptTemplate(
    input_variables=["title", "emotion"],
    template="""You need to write a powerful {emotion} speech of 350 words 
     for the following title: {title}.
     I dont want the title description.
     emotion description is Describe the given emotion {emotion}
     Format the output with 2 keys : 'title','speech','emotion' as emotion description and fill them with respective values
    """
)


def generate_speech(input):
    speech_response = mistral_client.chat.complete(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": input}],
    )
    return speech_response.choices[0].message.content


# simple sequential chain
first_chain = title_template | LLM1 | StrOutputParser() | (lambda title: (st.write(title),title)[1])
second_chain = RunnableLambda(lambda title: (speech_template.format(title=title,emotion=emotion))) | RunnableLambda(generate_speech) | JsonOutputParser()
final_chain = first_chain | second_chain

# first_chain = title_template | LLM1 | StrOutputParser() | (lambda title: (st.write(title), {"formated_speech": speech_template.format(title=title,emotion=emotion)})[1])
# second_chain = RunnableLambda(generate_speech) | JsonOutputParser()
# final_chain = first_chain | second_chain

topic = st.text_input("Enter the topic: ")
emotion = st.text_input("Enter the emotion: ")

if topic:
    response = final_chain.invoke({"topic": topic})
    st.write(response)
