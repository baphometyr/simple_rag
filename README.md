# Simple RAG

This project is part of my **portfolio** and aims to showcase a simple yet extensible Retrieval-Augmented Generation (RAG) implementation. While it is primarily for demonstration purposes, it can also serve as a starting point for anyone interested in building and deploying their own RAG pipelines.

## Features

* **Plug-and-play RAG system** with configurable components.
* **Streamlit web demo** to interact with collections and LLMs.
* Support for **CSV and PDF document ingestion**.
* Configurable **chunk splitting strategies**.
* Embedding and indexing with **FAISS** (more backends planned).
* Integration with **Hugging Face local models**, **OpenAI**, and **Azure OpenAI** (additional providers coming soon).
* Customizable generation parameters (temperature, number of retrieved documents, reranker option).

## Architecture Overview

The project follows a modular design:

1. **Data ingestion**: Upload CSV or PDF files.
2. **Chunking**: Split text into chunks with user-defined settings.
3. **Embedding**: Convert text chunks into vector representations.
4. **Indexing**: Store embeddings in FAISS (future: more index types).
5. **Retrieval**: Query the index to find relevant documents.
6. **Reranking (optional)**: Reorder retrieved documents with a cross-encoder reranker.
7. **LLM Generation**: Use OpenAI, Azure OpenAI, or Hugging Face models to answer queries.

```
User Query -> Retriever -> (Optional) Reranker -> LLM -> Final Answer
```

## Installation

Clone the repository and install the package:

```bash
git clone https://github.com/baphometyr/simple_rag
cd simple_rag
pip install -e .
```

### Optional dependencies

* For **Streamlit demo**:

```bash
pip install .[streamlit]
```

* For **development and testing**:

```bash
pip install .[dev]
```

## Running the Streamlit Demo

The project includes a Streamlit-based web interface for testing.

```bash
streamlit run streamlit/app.py
```

### Demo Features

* Load a sample **movie dataset** ([Kaggle link here](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset/data?select=keywords.csv)).
* Add your own collections by uploading CSV or PDF files.
* Choose embeddings, chunking strategies, and index type.
* Select LLM provider (OpenAI, Azure OpenAI, Hugging Face local).
* Adjust generation parameters (temperature, number of docs to retrieve, reranker on/off).

## Planned Features

* Support for additional vector databases (e.g., **ChromaDB**, **Weaviate**, **Pinecone**).
* Expanded LLM integrations (Anthropic, Cohere, etc.).
* More chunking and preprocessing strategies.
* Improved Streamlit UI/UX.

## Disclaimer

This is a **portfolio project**. While it can be useful for experimenting with RAG systems, it is not intended as a production-ready solution.
