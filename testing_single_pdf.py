import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

file_path = "/Users/mominagull/Projects/rag_literature_review/data/Evaluation of the reliability of resistance spot welding con.pdf"

def chunk_and_embed_pdf(pdf_path, openai_api_key, chunk_size=2000, chunk_overlap=200):
    """Chunk and embed a PDF file using OpenAI's embeddings.

    Args:
        pdf_path (str): The path to the PDF file.
        openai_api_key (str): OpenAI API key.
        chunk_size (int): The number of characters in each chunk.
        chunk_overlap (int): The number of characters to overlap between chunks.

    Returns:
        FAISS: A FAISS VectorStore containing the embedded chunks.
    """
    # Load the PDF
    loader = PyPDFLoader(pdf_path)
    pages = [page for page in loader.lazy_load()]

    # Split the text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                              chunk_overlap=chunk_overlap, 
                                              separators=["\n\n", "\n", " ", ""])
    
    docs = []
    for page in pages:
        chunks = splitter.split_text(page.page_content)

        # Convert each chunk into a "Document" object for LangChain
        for chunk in chunks:
            docs.append({
                "page_content": chunk,
                "metadata": {
                    "page_number": page.metadata.get("page"),
                    "source": pdf_path
                }
            })

    # Embed the chunks
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)

    # Store them in a FAISS VectorStore
    # FAISS expects a list of strings or a list of Document objects
    # We'll pass the chunk text as a list, and keep metadata in a parallel structure.
    texts = [doc["page_content"] for doc in docs]
    metadatas = [doc["metadata"] for doc in docs]

    # Build the vectorstore
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)

    return vectorstore


if __name__ == "__main__":
    
    pdf_file = "/Users/mominagull/Projects/rag_literature_review/data/Evaluation of the reliability of resistance spot welding con.pdf"
    openai_key = os.environ.get("OPENAI_API_KEY")

    # Create the VectorStore from the PDF
    faiss_store = chunk_and_embed_pdf(pdf_file, openai_key)

    # Quick test: Query the VectorStore
    query = "What is the main focus of the paper?"
    docs = faiss_store.similarity_search(query, k=3)

    print("Top 3 Relevant Chunks:")
    for i, doc in enumerate(docs, start=1):
        print(f"\n--- Chunk #{i} ---")
        print("Content:", doc.page_content)
        print("Metadata:", doc.metadata)