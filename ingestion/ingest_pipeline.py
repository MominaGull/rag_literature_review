import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from vectorstore.pinecone_index import get_or_create_pinecone_index
# from vectorstore.milvus_index import get_or_create_milvus_collection
from ingestion.embedding import get_openai_embeddings, embed_documents
from ingestion.chunking import chunk_documents
# from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document
# from ingestion.markdown_parser import parse_all_markdowns
from ingestion.pdf_parser import parse_all_pdfs, parse_create_markdown
# from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv(override=True)
openai_key = os.environ.get("OPENAI_API_KEY")


def main(pdf_dir=None):

    # 1. Initialize Pinecone
    index = get_or_create_pinecone_index()

    # 2. Parse all PDFs
    # data = parse_all_pdfs()

    # 2. Parse all markdowns
    # data = parse_all_markdowns()

    filenames, data = parse_create_markdown(pdf_dir)
    # print("data:", data)

    # markdown_path = os.path.join("/Users/mominagull/Projects/rag_literature_review/markdown_data/A new measurement method for the dynamic resistance signal during the resistance spot welding process.md")
    # loader = UnstructuredMarkdownLoader(markdown_path)
    # data = loader.load()
    # print(data)
    # print(data[0].page_content)

    for filename in filenames:
        print(f"Filename: {filename}")
        print(f"Data: {data[filename]}")

        # Create a Document object directly
        document = Document(
            page_content=data[filename],
            metadata={"source": filename, "format": "markdown"}
        )

        # 3. Chunk each PDF's pages
        all_chunked_docs = chunk_documents(
            document)

        # 4. Embed the chunked docs
        embeddings = get_openai_embeddings(openai_key)
        upserts = embed_documents(all_chunked_docs, embeddings, filename)
        print(upserts)

    # vector_store = get_or_create_milvus_collection(embeddings)
    # Index chunks
    # response = vector_store.upsert(upserts)
    # print(response)

        # 5. Upsert into Pinecone
        # upserts is list of (id, vector, metadata)
        index.upsert(vectors=upserts)

    print(
        f"Successfully ingested {len(upserts)} chunks")


if __name__ == "__main__":
    main()
