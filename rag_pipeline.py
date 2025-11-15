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
            vector_database: str="./data/database/embeddings.npy", 
            docs_database: str="./data/database/documents.pkl",
            llm_model: str="meta-llama/Llama-2-7b-chat-hf",
            embedding_model: str="all-MiniLM-L6-v2",
            cross_encoder_model: str='cross-encoder/ms-marco-MiniLM-L-6-v2',
            top_k: int=20,
            top_k_rerank: int=3
        ):
        """
        Initialize RAG Pipeline with vector DB and LLM.
        Args:
            vector_database: Path to the vector database (numpy file).
            docs_database: Path to the documents database (pickle file).
            llm_model: Pretrained LLM model name or path.
            embedding_model: Pretrained embedding model name or path.
            cross_encoder_model: Pretrained cross-encoder model for reranking.
            top_k: Number of top documents to retrieve - before reranking.
            top_k_rerank: Number of top documents to rerank and use for context.
        """
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

    def get_top_k_docs(self, query: str, method: str='cosine') -> tuple:
        """
        Retrieve top-k documents from the vector database.
        Args:
            query: Input query string.
            method: Retrieval method - 'cosine' or 'ann'.
        Returns:
            top_k_docs: List of top-k retrieved document contents.
            indices: Indices of the retrieved documents in the database.
        """
        if method == 'cosine':
            indices = self.get_top_k_docs_cosine(query)
        elif method == 'ann':
            indices = self.get_top_k_docs_ann(query)
        else:
            raise ValueError(f"Unknown method: {method}")

        top_k_docs = [self.docs[i]['page_content'] for i in indices]
        return top_k_docs, indices

    def get_top_k_docs_cosine(self, query: str) -> list:
        """
        Retrieve top-k documents using cosine similarity.
        Args:
            query: Input query string.
        Returns:
            indices: Indices of the retrieved documents in the database.
        """
        query_embedding = self.embedding_model.encode(query)
        sims = cosine_similarity([query_embedding], self.embeddings)[0]
        indices = np.argsort(sims)[-self.top_k:][::-1]
        return indices

    def get_top_k_docs_ann(self, query: str) -> list:
        """
        Retrieve top-k documents using approximate nearest neighbors.
        Args:
            query: Input query string.
        Returns:
            indices: Indices of the retrieved documents in the database.
        """
        query_embedding = self.embedding_model.encode(query)
        ann = NearestNeighbors(n_neighbors=self.top_k, algorithm='ball_tree')
        ann.fit(self.embeddings)
        _distances, indices = ann.kneighbors([query_embedding])
        return indices[0]

    def rerank_docs(self, query: str, context: list) -> list:
        """
        Rerank retrieved documents using a cross-encoder.
        Args:
            query: Input query string.
            context: List of retrieved document contents.
        Returns:
            reranked_indices: Indices of the reranked documents.
        """
        pairs = [[query, doc] for doc in context]
        scores = self.reranker.predict(pairs)
        reranked_indices = np.argsort(scores)[-self.top_k_rerank:][::-1]
        return reranked_indices

    def generate_answer(self, query: str, method: str="ann") -> tuple:
        """
        Generate an answer to the query using the RAG pipeline.
        Args:
            query: Input query string.
            method: Retrieval method - 'cosine' or 'ann'.
        Returns:
            answer: Generated answer string.
        """
        context, _ = self.get_top_k_docs(query, method=method)
        reranked_indices = self.rerank_docs(query, context)
        context = [context[i] for i in reranked_indices]

        context_str = "\n\n".join(context)
        prompt = f"Context: {context_str}\n\nQuestion: {query}\n\nAnswer:"
        response = self.llm(prompt, max_length=1024, temperature=0.7)
        return response[0]["generated_text"]
