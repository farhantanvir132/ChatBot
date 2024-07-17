
import os
os.environ['HF_TOKEN']="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"
os.environ['HUGGINGFACEHUB_API_TOKEN']="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"
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from time import time
import pandas as pd
from langchain.document_loaders import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline

# Load your CSV data
loader = CSVLoader("myexel.csv", encoding="utf-8")
documents = loader.load()

# Initialize HuggingFace pipeline for text generation
model_id = 'Undi95/Meta-Llama-3-8B-hf'
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
query_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Initialize LLM for retrieval-based QA
llm = HuggingFacePipeline.from_pretrained(model_id)

# Initialize embeddings and vector database for retrieval-based QA
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cuda"}
embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
vectordb = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory="chroma_db")
retriever = vectordb.as_retriever()
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, verbose=True)

# Create a list to store chat history
chat_history = []

# Function to generate response to user query
def generate_response(query):
    # Add user query to chat history
    chat_history.append(("User", query))
    # Get response using RAG model
    response = qa.run(query)
    # Add response to chat history
    chat_history.append(("System", response))
    return response

# Streamlit app
st.title("Chat with Meta Llama 3")
query_input = st.text_input("Enter your query:")

if st.button("Send"):
    if query_input.strip() != "":
        # Generate response to user query
        response = generate_response(query_input)
        # Display response
        st.text_area("Response:", value=response, height=100)

# Display chat history
st.subheader("Chat History")
for role, message in chat_history:
    st.text(f"{role}: {message}")


