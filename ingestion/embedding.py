import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document

def get_openai_embeddings(api_key: str):
    return OpenAIEmbeddings(openai_api_key=api_key)

def embed_documents(docs: list[Document], 
                    embeddings) -> list[tuple[str, list[float], dict]]:
    """
    Embeds a list of documents and prepares them for vector database upsert.
    
    Args:
        docs (list[Document]): List of Document objects to embed
        embeddings: Embedding model instance (e.g., OpenAIEmbeddings)
        
    Returns:
        list[tuple[str, list[float], dict]]: List of tuples containing:
            - str: Unique ID for each document chunk
            - list[float]: Embedding vector for the chunk
    """
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]

    vectors = embeddings.embed_documents(texts)

    # Construct unique IDs for each chunk
    upserts = []
    for i, vector in enumerate(vectors):
        chunk_id = f"doc-chunk-{i}"
        upserts.append(
            (chunk_id, vector, metadatas[i])
        )
    return upserts
