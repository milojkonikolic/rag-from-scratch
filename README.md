# RAG from Scratch
An end-to-end RAG pipeline built from scratch, including database creation, text chunking, embedding generation, vector search, and answer retrieval.

## Overview

The project outlines RAG implementation consisting of the following:
- Creation of simple local database: Reading PDFs, chunking (creating documents), and storing embedding of chunks in database (.npy file)
- RAG Pipeline:
  - Creation of embedding for a given query
  - Measuring semantic similarity between query and embeddings from databse
  - Retrieval of top k documents based on similarity score
  - Reranking using cross-encoder that creates final list of documents
  - Answer generation using LLM and retrievaed documents as a context
- Implementation of several metrics: recall, precision, average precision, reciprocal rank, etc.

## Installation
Clone the repository:
```
git clone git@github.com:milojkonikolic/rag-from-scratch.git
```
Install required dependencies inside clonned repository:
```
pip install -r requirements.txt
```

## Database

Database is created from 18 business books (PDF files) downloaded from this [web page](https://openstax.org/subjects/business).

PDF files are divided first into pages, and then into chunks of the fixed size = 1000 characters (overlap is 100). Database consists of 11461 pages and 39051 documents (chunks). Raw documents are saved to `./data/database/documents.pkl` and embedding of documents is stored to `./data/database/embeddings.npy` (vector database) in order to provide fast access to embedding vectors of original documents.

## How it Works

The pipeline works by running the script `answer_generation.py` and providing the query. For the best results, the query should be related to the topics mentioned in the books that are used to create database. Query is first transformed to embedding vector using the same model that is used to create vector database. Embedding of the query is than compared to embedding vectors saved in database, and the similarity score is measured for each doc from database. Documents with highest scores (20 docs by default) are returned. Then reranker is used to create final list of docs that will be used as context. Cross-encoder model is used to rerank retrieved documents and create final 3 documents. Reranker works by merging retrieved documents and query and then measuring similarity score. Three docs with the highest score are returned and sent to LLM for inference. Default LLM model is `Llama-2-7b`. The LLM, using original query and retrieved documents as context generates the answer.

The main point of this RAG pipeline is to get answers from some bussines books instead from some general knowledge from LLM. As a result, the answers should be more accurate and relevant.

### Examples
Here is an example of the answer for the given query (question below).
```
Question: What is statement of owner’s equity?

Answer: The statement of owner's equity is a financial statement that summarizes the changes in a company's owner's equity over a period of time. It shows the net worth or value of the business at the end of the period, and highlights changes in the ownership structure through issuance or repurchase of shares, dividends, and other transactions. The statement of owner's equity is an important tool for investors, creditors, and other stakeholders to evaluate the financial health and performance of a company.

```

## License
rag-from-scratch project is licensed under the MIT License.
