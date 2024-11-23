#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 15:15:09 2024

@author: komal
"""

import os
import chromadb
import streamlit as st
from dotenv import load_dotenv
from google.api_core import retry
import google.generativeai as genai
from langchain.document_loaders import UnstructuredURLLoader
from chromadb import Documents, EmbeddingFunction, Embeddings

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

class GeminiEmbeddingFunction(EmbeddingFunction):
    
    # Specify whether to generate embeddings for documents, or queries
    document_mode = True

    def __call__(self, input: Documents) -> Embeddings:
        if self.document_mode:
            embedding_task = "retrieval_document"
        else:
            embedding_task = "retrieval_query"

        retry_policy = {"retry": retry.Retry(predicate=retry.if_transient_error)}

        response = genai.embed_content(
            model="models/text-embedding-004",
            content=input,
            task_type=embedding_task,
            request_options=retry_policy,
        )
        return response["embedding"]
    
st.title("News Research Tool")

news_urls = []
st.sidebar.itle("News Article URLs")
for i in range(3):
    url = st.sidebar.text_input(f"URL: {i+1}")
    news_urls.append(url)
    
process_url_clicked = st.sidebar.button("Process URLs")
if process_url_clicked:
    loader = UnstructuredURLLoader(urls=news_urls)
    data = loader.load()
    documents = []
    metadatas = []
    for chunk in data:
      documents.append(chunk.page_content)
      metadatas.append(chunk.metadata)
      
    DB_NAME = "newsarticledb"
    embed_fn = GeminiEmbeddingFunction()
    embed_fn.document_mode = True
    chroma_client = chromadb.Client()
    db = chroma_client.get_or_create_collection(name=DB_NAME, embedding_function=embed_fn)
    db.add(documents=documents, metadatas=metadatas, ids=[str(i) for i in range(len(documents))])
     
    
    

    
news_urls = ['https://finance.yahoo.com/news/apple-dan-riccio-key-executive-184819662.html', \
              'https://www.cnbc.com/2024/10/09/nvidia-stock-up-25percent-in-a-month-as-stock-closes-in-on-new-record.html',\
              'https://www.bloomberg.com/news/articles/2024-10-09/apple-s-dan-riccio-key-executive-in-both-the-jobs-and-cook-eras-to-retire']

loader = UnstructuredURLLoader(urls=news_urls)
data = loader.load()

documents = []
metadatas = []
for chunk in data:
  documents.append(chunk.page_content)
  metadatas.append(chunk.metadata)
  
DB_NAME = "newsarticledb"
embed_fn = GeminiEmbeddingFunction()
embed_fn.document_mode = True

chroma_client = chromadb.Client()
db = chroma_client.get_or_create_collection(name=DB_NAME, embedding_function=embed_fn)

db.add(documents=documents, metadatas=metadatas, ids=[str(i) for i in range(len(documents))])

# Switch to query mode when generating embeddings.
embed_fn.document_mode = False

# Search the Chroma DB using the specified query.
query = "What was the close price of NVIDIA?"

result = db.query(query_texts=[query], n_results=1)
[[passage]] = result["documents"]

passage_oneline = passage.replace("\n", " ")
query_oneline = query.replace("\n", " ")

# This prompt is where you can specify any guidance on tone, or what topics the model should stick to, or avoid.
prompt = f"""You are a helpful and informative bot that answers questions using text from the reference passage included below.
Be sure to respond in a complete sentence, being comprehensive, including all relevant background information.
However, you are talking to a non-technical audience, so be sure to break down complicated concepts and
strike a friendly and converstional tone. If the passage is irrelevant to the answer, you may ignore it.

QUESTION: {query_oneline}
PASSAGE: {passage_oneline}
"""
print(prompt)

model = genai.GenerativeModel("gemini-1.5-flash-latest")
answer = model.generate_content(prompt)