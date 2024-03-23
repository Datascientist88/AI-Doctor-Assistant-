import os
from dotenv import load_dotenv
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.vectorstores.qdrant import Qdrant
import qdrant_client
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
import tiktoken
from langchain.chains.combine_documents import create_stuff_documents_chain

# load the variables
load_dotenv()

collection_name = os.getenv("QDRANT_COLLECTION_NAME")


# get the vector stor
def get_vector_store():
    client = qdrant_client.QdrantClient(
        url=os.getenv("QDRANT_HOST"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    embeddings = OpenAIEmbeddings()
    vector_store = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embeddings,
    )
    return vector_store


vector_store = get_vector_store()


def get_context_retriever_chain(vector_store=vector_store):
    llm = ChatOpenAI()
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            (
                "user",
                "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation",
            ),
        ]
    )
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain


def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a specialized doctor AI medical assistant programmed to address inquiries about medicine, ICD10 codes, diagnosis,, symptoms  and differential diagnosis in form of list , enumerate them please when asked about differential diagnosis and symptoms of a disease enumerate them . 
                Your responses should strictly adhere to the medical field context:\n\n{context} you have been trained in. Avoid providing general knowledge answers or responses outside of your medical training. 
                If a question falls outside of the medical realm or exceeds your expertise, reply with: Sorry, I don't know about this as it's beyond my training context as a medical AI assistant. 
                Refrain from answering queries on unrelated topics such as religions, sports, programming, and others listed here 
                [ religions, general knowledge , sports ,non-medical sciences ,
                universe,math , programming, coding, outfits , cultures, ethnicities, Management ,
                business , politics , how to  make something like food, agriculture all general knowledge topics except medicine,..... etc ], as they lie outside your scope of expertise be polite and recognize greetings like hi , hello etc.
                """
          ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ]
    )
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    response_stream = conversation_rag_chain.stream(
        {"chat_history": st.session_state.chat_history, "input": user_input}
    )
    for chunk in response_stream:
        content=chunk.get("answer","")
        yield content


# app layout
st.set_page_config("Conversational AI Doctor ", "🤖")
st.title("Doctor AI Assistant 👨‍⚕️")
with st.sidebar:
    photo_url = "https://i.ibb.co/3k14LmY/Whats-App-Image-2024-02-10-at-9-03-47-AM.jpg"

# Add HTML to the sidebar to display the image as a circle
    st.markdown(
        f'<a href="https://ibb.co/6NYrf0J"><img src="{photo_url}" alt="Your Photo" style="width: 100px; height: 100px; border-radius: 50%;"></a>',
        unsafe_allow_html=True
    )
    st.markdown(
    "<div style='text-align: justify'>"
    "The Doctor AI Assistant is an advanced artificial intelligence tool designed to aid physicians in diagnosing diseases swiftly and accurately. "
    "It provides comprehensive support by addressing a wide array of queries, including those related to ICD10 codes, diagnoses, symptoms, and differential diagnoses across all medical specialties. "
    "Additionally, it assists in the submission of relevant insurance claims and ensures adherence to drug indications consistent with ICD10 codes, guidelines, and best medical practices. "
    "With multilingual capabilities, it offers assistance in all languages spoken worldwide, empowering healthcare professionals with unparalleled efficiency and accuracy in patient care. "
    "This AI App was developed by <b>MOHAMMED BAHAGEEL</b>, Artificial intelligence scientist as a part of his experiments using Retrieval Augmented Generation."
    "</div>",
    unsafe_allow_html=True
)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content=" Hello ! with you is Doctor Assistant AI  chatbot  how can I assist you today  with your medical questions ? 🥰")]
if "vector_store" not in st.session_state:
    st.session_state.vector_store = get_vector_store()
for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI", avatar="🤖"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human", avatar="👨‍⚕️"):
                st.write(message.content)
# user input
user_query = st.chat_input("Type your message here...")
#response=get_response(user_query)
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("Human", avatar="👨‍⚕️"):
        st.markdown(user_query)
    with st.chat_message("AI",avatar="🤖"):
        response=st.write_stream(get_response(user_query))
    st.session_state.chat_history.append(AIMessage(content=response))
