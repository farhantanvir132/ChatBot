
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import CSVLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_groq import ChatGroq
import streamlit as st
from dotenv import load_dotenv
import time
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import PyPDFDirectoryLoader
load_dotenv()

## Load the GROQ API KEY
os.environ['HUGGINGFACEHUB_API_TOKEN']="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"
groq_api_key = os.getenv('GROQ_API_KEY')

st.title("ChatBot-THE VoG")

llm = ChatGroq(groq_api_key=groq_api_key,
                model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
</context>
<context>
Questions:{input}
</context>
"""
)

def vector_embedding():
    if "vectors" not in st.session_state:

       st.session_state.loader = PyPDFDirectoryLoader("./datas")  ## Data Ingestion
       st.session_state.docs = st.session_state.loader.load()  ## Document Loading
       st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
       model_name = "sentence-transformers/all-mpnet-base-v2"
       st.session_state.embeddings = HuggingFaceEmbeddings(model_name=model_name)
       st.session_state.vectors = Chroma.from_documents(documents=st.session_state.text_splitter.split_documents(st.session_state.docs), embedding=st.session_state.embeddings, persist_directory="chroma_db")  # Store vectordb in session state for later use
       st.write("Vog is Ready to answer your question")

vector_embedding()
if "history" not in st.session_state:
    st.session_state.history=[]
for message in st.session_state.history:
    if isinstance(message,HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)
prompt1 = st.chat_input("Enter Your Question From Documents")
if prompt1:
    st.session_state.history.append(HumanMessage(prompt1))
    if "vectors" in st.session_state:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        print("Response time :", time.process_time() - start)
        with st.chat_message("Human"):
            st.markdown(prompt1)
        with st.chat_message("AI"):
            st.markdown(response['answer'])
        st.session_state.history.append(AIMessage(response['answer']))
    else:
        st.error("Vector Store is not initialized. Please run 'Documents Embedding' first.")



