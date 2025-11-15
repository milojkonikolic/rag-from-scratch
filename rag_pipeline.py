import numpy as np
import pickle
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import CrossEncoder


class RAGPipeline():
    def __init__(
            self, 
            vector_database="./data/database/embeddings.npy", 
            docs_database="./data/database/documents.pkl",
            llm_model="meta-llama/Llama-2-7b-chat-hf",
            embedding_model="all-MiniLM-L6-v2",
            cross_encoder_model='cross-encoder/ms-marco-MiniLM-L-6-v2',
            top_k=20,
            top_k_rerank=3
        ):
        self.embeddings = np.load(vector_database)
        with open(docs_database, 'rb') as f:
            self.docs = pickle.load(f)
        self.embedding_model = SentenceTransformer(embedding_model)
        self.llm = pipeline(
            "text-generation",
            model=llm_model,
            device_map="auto"
        )
        self.reranker = CrossEncoder(cross_encoder_model)
        self.top_k = top_k
        self.top_k_rerank = top_k_rerank

    def get_top_k_docs(self, query, method='cosine'):
        if method == 'cosine':
            indices = self.get_top_k_docs_cosine(query)
        elif method == 'ann':
            indices = self.get_top_k_docs_ann(query)
        else:
            raise ValueError(f"Unknown method: {method}")

        top_k_docs = [self.docs[i]['page_content'] for i in indices]
        return top_k_docs

    def get_top_k_docs_cosine(self, query):
        query_embedding = self.embedding_model.encode(query)
        sims = cosine_similarity([query_embedding], self.embeddings)[0]
        indices = np.argsort(sims)[-self.top_k:][::-1]
        return indices

    def get_top_k_docs_ann(self, query):
        query_embedding = self.embedding_model.encode(query)
        ann = NearestNeighbors(n_neighbors=self.top_k, algorithm='ball_tree')
        ann.fit(self.embeddings)
        _distances, indices = ann.kneighbors([query_embedding])
        return indices[0]

    def generate_answer(self, query, method="ann"):
        context = self.get_top_k_docs(query, method=method)

        pairs = [[query, doc] for doc in context]
        scores = self.reranker.predict(pairs)
        reranked_indices = np.argsort(scores)[-self.top_k_rerank:][::-1]
        context = [context[i] for i in reranked_indices]

        context_str = "\n\n".join(context)
        prompt = f"Context: {context_str}\n\nQuestion: {query}\n\nAnswer:"
        response = self.llm(prompt, max_length=1024, temperature=0.7)
        return response[0]["generated_text"]
