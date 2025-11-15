# RAG from Scratch
An end-to-end RAG pipeline built from scratch, including database creation, text chunking, embedding generation, vector search, and answer retrieval.

## Overview

This repository demonstrates a minimal RAG implementation covering:
- Database Creation:
  - Reading and parsing PDFs
  - Splitting content into pages and fixed-size text chunks
  - Generating embeddings for all chunks
  - Storing documents and their vectors in a local database (`.pkl` and `.npy` files)
- RAG Pipeline:
  - Creating of embedding for a given query
  - Computing semantic similarity between the query and stored embeddings
  - Retrieving top k most relevant chunks
  - Reranking retrieved documents using a cross-encoder
  - Answer generation using LLM and retrieved documents as a context
- Implementation of several metrics: recall, precision, average precision, reciprocal rank, etc.

## Installation
Clone the repository:
```
git clone https://github.com/milojkonikolic/rag-from-scratch.git
```
Install required dependencies inside cloned repository:
```
pip install -r requirements.txt
```

## Database

The vector database is built from 18 business-related books (PDF files) downloaded from this [web page](https://openstax.org/subjects/business).

PDF files are divided first into pages, and then into chunks of the fixed size (1000 characters where overlap is 100). Database consists of 11461 pages and 39051 documents (chunks). Raw documents are saved to `./data/database/documents.pkl` and embeddings of documents is stored to `./data/database/embeddings.npy` in order to provide fast access to embedding vectors of original documents.

## How it Works

The pipeline is executed by running the `answer_generation.py` script and providing a query. For best results, the query should align with the topics covered in the books used to build the database. The query is first encoded into an embedding vector using the same model applied during database construction. This vector is then compared against all stored embeddings to compute similarity scores. The top 20 most relevant documents are retrieved by default.

These candidates are then passed to a reranker, which uses a cross-encoder to evaluate each (query, document) pair and produce a relevance score. The top three documents from this reranking step are selected as the final context for answer generation. The default LLM model, `Llama-2-7b-chat`, receives both the original query and the selected documents and generates the final answer grounded in the retrieved content.

The primary goal of this RAG pipeline is to generate answers grounded in the curated set of business books, rather than relying solely on the LLM’s general knowledge. This ensures that the responses are more accurate and domain-specific.

### Examples
Here is an example of the answer for the given query (question below).
```
Question: What is statement of owner’s equity?

Answer: The statement of owner's equity is a financial statement that summarizes the changes in a company's owner's equity over a period of time. It shows the net worth or value of the business at the end of the period, and highlights changes in the ownership structure through issuance or repurchase of shares, dividends, and other transactions. The statement of owner's equity is an important tool for investors, creditors, and other stakeholders to evaluate the financial health and performance of a company.

```

## License
rag-from-scratch project is licensed under the MIT License.
