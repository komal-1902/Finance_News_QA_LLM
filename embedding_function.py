#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 16:27:11 2024

@author: komal
"""

import os
from dotenv import load_dotenv
from google.api_core import retry
import google.generativeai as genai
from chromadb.utils.embedding_functions import EmbeddingFunction

# Configure Generative AI
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

class GeminiEmbeddingFunction(EmbeddingFunction):
    """
    Embedding function using Google Gemini.
    """
    document_mode = True

    def __call__(self, input):
        task_type = "retrieval_document" if self.document_mode else "retrieval_query"
        retry_policy = {"retry": retry.Retry(predicate=retry.if_transient_error)}
        response = genai.embed_content(
            model="models/text-embedding-004",
            content=input,
            task_type=task_type,
            request_options=retry_policy,
        )
        return response["embedding"]
