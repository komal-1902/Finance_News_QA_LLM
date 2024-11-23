#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 16:29:16 2024

@author: komal
"""

import os
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai
from embedding_function import GeminiEmbeddingFunction
from langchain.document_loaders import UnstructuredURLLoader
from db_utils import initialize_chroma_db, add_documents_to_db, url_exists_in_db


# Configure Generative AI
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize DB and embedding function
DB_PATH = "vector_database"
DB_NAME = "newsarticledb"
embed_fn = GeminiEmbeddingFunction()

# Initialize or load database
db, chroma_client = initialize_chroma_db(DB_PATH, DB_NAME, embed_fn)

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
            if url_exists_in_db(db, source_url):
                st.sidebar.warning(f"URL already in database: {source_url}")
                continue
            documents.append(chunk.page_content)
            metadatas.append(chunk.metadata)
        
        # Add documents to database
        db_count = add_documents_to_db(db, documents, metadatas)
        st.sidebar.success(f"URLs processed and added to the database! \n We currently have {str(db_count)} articles.: ")

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
    
    model = genai.GenerativeModel("gemini-1.5-flash-latest")
    prompt = f"""
    QUESTION: {query_oneline}
    PASSAGE: {passage_oneline}
    """
    answer = model.generate_content(prompt)
    st.write(answer.text)
    st.write("Source: " + source_url['source'])
