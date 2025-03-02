import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

from ingestion.pdf_parser import parse_all_pdfs
from ingestion.chunking import chunk_documents
from ingestion.embedding import get_openai_embeddings, embed_documents
from vectorstore.pinecone_index import get_or_create_pinecone_index

def main():

    openai_key = os.environ.get("OPENAI_API_KEY")

    # 1. Initialize Pinecone
    index = get_or_create_pinecone_index()

    # 2. Parse all PDFs
    pdfs_data = parse_all_pdfs() # Returns list of (filename, list_of_page_docs)
    
    # 3. Chunk each PDF's pages
    all_chunked_docs = []
    for (filename, pages) in pdfs_data:
        chunked = chunk_documents(pages, chunk_size=1000, chunk_overlap=100)
        all_chunked_docs.extend(chunked)

    # 4. Embed the chunked docs
    embeddings = get_openai_embeddings(openai_key)
    upserts = embed_documents(all_chunked_docs, embeddings)
    
    # 5. Upsert into Pinecone
    # upserts is list of (id, vector, metadata)
    index.upsert(vectors=upserts)

    print(f"Successfully ingested {len(upserts)} chunks across {len(pdfs_data)} PDFs.")

if __name__ == "__main__":
    main()
