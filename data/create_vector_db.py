import PyPDF2
import numpy as np
import pickle
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


# Load and process documents
def load_pdf_documents(directory):
    docs = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.pdf'):
                filepath = os.path.join(root, file)
                with open(filepath, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    for page_num, page in enumerate(reader.pages):
                        text = page.extract_text()
                        docs.append({'content': text, 'source': filepath, 'page': page_num})
    return docs

def chunk_documents(documents, chunk_size=1000, chunk_overlap=100):
    chunks = []
    for doc in tqdm(documents, desc="Chunking documents"):
        text = doc['content']
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                chunks.append({
                    'page_content': chunk,
                    'metadata': {'source': doc['source'], 'page': doc['page']}
                })
    return chunks


if __name__ == "__main__":

    print("Loading and processing documents...")
    documents = load_pdf_documents('/hdd/Datasets/openstax-books/economy/')
    print(f"Loaded {len(documents)} documents")
    docs = chunk_documents(documents, chunk_size=1000, chunk_overlap=100)
    print(f"Chunked into {len(docs)} document chunks")

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Generate embeddings for all documents
    embedding_list = [embedding_model.encode_query(doc['page_content']) for doc in docs]
    embedding_array = np.array(embedding_list)

    # Save vector DB
    os.makedirs('./data/database', exist_ok=True)
    np.save('./data/database/embeddings.npy', embedding_array)
    with open('./data/database/documents.pkl', 'wb') as f:
        pickle.dump(docs, f)

