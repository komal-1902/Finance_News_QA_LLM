#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 16:29:16 2024

@author: komal
"""
__import__("pysqlite3")
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import nltk 
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

import os
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai
from embedding_function import GeminiEmbeddingFunction
from langchain.document_loaders import UnstructuredURLLoader
from db_utils import initialize_chroma_db, add_documents_to_db, list_urls_in_db


# Configure Generative AI
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize DB and embedding function
DB_PATH = os.getenv("DB_PATH")
DB_NAME = os.getenv("DB_NAME")
embed_fn = GeminiEmbeddingFunction()

# Initialize or load database
db, chroma_client = initialize_chroma_db(DB_PATH, DB_NAME, embed_fn)

urls_in_db = list_urls_in_db(db)
print("Articles in Database:")
print(urls_in_db)

st.title("News Research Tool")

# Sidebar for URLs
st.sidebar.title("News Article URLs")
news_urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    news_urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

if process_url_clicked:
    
    # Load data from URLs
    loader = UnstructuredURLLoader(urls=[url for url in news_urls if url])
    data = loader.load()
    
    if data == []:
        st.sidebar.text("Add some articles")
    else:
    
        documents, metadatas = [], []
        for chunk in data:
            source_url = chunk.metadata.get("source_url")
            if source_url in urls_in_db:
                st.sidebar.warning(f"URL already in database: {source_url}")
                continue
            documents.append(chunk.page_content)
            metadatas.append(chunk.metadata)
        
        if documents and metadatas:
            # Add documents to database
            db_count = add_documents_to_db(db, documents, metadatas)
            st.sidebar.success(
                f"URLs processed and added to the database! "
                f"\nWe currently have {db_count} articles."
            )
        else:
            st.sidebar.warning("No new articles were added to the database.")

# Query input
query = st.text_input("Enter your query:")
if query:
    
    # Switch to query mode
    embed_fn.document_mode = False
    
    # Query the database
    result = db.query(query_texts=[query], n_results=1)
    [[passage]] = result["documents"]
    [[source_url]] = result["metadatas"]
    
    # Display the passage and generate an answer
    passage_oneline = passage.replace("\n", " ")
    query_oneline = query.replace("\n", " ")
    
    #model = genai.GenerativeModel("gemini-1.5-flash-latest")
    model = genai.GenerativeModel("gemini-pro")
    prompt = f"""
    QUESTION: {query_oneline}
    PASSAGE: {passage_oneline}
    """
    answer = model.generate_content(prompt)
    st.write(answer.text)
    st.write("Source: " + source_url['source'])
