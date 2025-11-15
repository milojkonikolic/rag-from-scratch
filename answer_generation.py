from rag_pipeline import RAGPipeline


if __name__ == "__main__":
    
    rag_pipeline = RAGPipeline()
    query = "What are White collar crimes?"

    answer = rag_pipeline.generate_answer(query)

    print("\nGenerated Answer via RAGPipeline:")
    print(answer)
