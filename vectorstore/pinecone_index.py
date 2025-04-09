# vectorstore/pinecone_index.py
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv(override=True)

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")
PINECONE_ENV = os.environ.get("PINECONE_ENV", "us-east-1")
EMBED_DIM = 1536

def get_or_create_pinecone_index():
    """
    Initialize Pinecone, create the index if it doesn't exist,
    then return a reference to the Pinecone Index object.
    """
    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Check if the index exists
    existing_indexes = pc.list_indexes().names()
    if PINECONE_INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=PINECONE_ENV
            )
        )

    return pc.Index(PINECONE_INDEX_NAME)
