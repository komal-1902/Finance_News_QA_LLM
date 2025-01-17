#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 16:25:21 2024

@author: komal
"""

import chromadb
from chromadb.config import Settings

def initialize_chroma_db(db_path, db_name, embedding_function):
    """
    Initialize ChromaDB with persistence support.
    """
    #client = chromadb.Client(Settings(persist_directory=db_path))
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name=db_name, embedding_function=embedding_function)
    return collection, client

def list_urls_in_db(collection):
    """
    Retrieve and print all URLs currently stored in the database.
    """
    try:
        all_metadatas = collection.get(include=["metadatas"])["metadatas"]
        urls = [metadata.get("source") for metadata in all_metadatas if metadata and "source" in metadata]
        
        return urls
    except KeyError:
        print("No URLs found in the database.")
        return []

def add_documents_to_db(collection, documents, metadatas, overwrite=True):
    """
    Add new documents toxw the database. Optionally overwrite existing content.
    """
    current_count = collection.count()
    ids = [str(i + current_count) for i in range(len(documents))]
    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print("Added URLS: ")
    print(metadatas)
    print("\nURLs in database currently:")
    print(list_urls_in_db(collection))
    return collection.count()

def url_exists_in_db(collection, url):
    """
    Check if a given URL already exists in the database.
    """
    # Get all metadata from the collection
    all_metadatas = collection.get(include=["metadatas"])["metadatas"]

    # Check if the URL is already in the metadata
    return any(metadata.get("source") == url for metadata in all_metadatas)