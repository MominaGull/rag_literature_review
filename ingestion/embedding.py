# import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document

def get_openai_embeddings(api_key: str):
    return OpenAIEmbeddings(openai_api_key=api_key)

def embed_documents(docs: list[Document], 
                    embeddings, filename) -> list[tuple[str, list[float], dict]]:

    """
    Embeds the documents using OpenAI embeddings.
    Args:
        doc (Document): Document object to embed
        embeddings (OpenAIEmbeddings): OpenAI embeddings instance
    Returns:
        list[tuple[str, list[float], dict]]: List of tuples containing (id, vector, metadata)
    """
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]

    vectors = embeddings.embed_documents(texts)

    # Construct unique IDs for each chunk
    upserts = []
    for i, vector in enumerate(vectors):
        # print(i, vector)
        chunk_id = f"{metadatas[i]['source']}-chunk-{i}"
        upserts.append(
            {"id": chunk_id, "values": vector, "metadata": metadatas[i]}
        )
    return upserts
