import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

from pinecone import Pinecone, ServerlessSpec


import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

def chunk_and_embed_pdf(pdf_path, openai_api_key, index_name, chunk_size=1000, chunk_overlap=100):
    """Chunk and embed a PDF file using OpenAI's embeddings.

    Args:
        pdf_path (str): The path to the PDF file.
        openai_api_key (str): OpenAI API key.
        chunk_size (int): The number of characters in each chunk.
        chunk_overlap (int): The number of characters to overlap between chunks.

    Returns:
        FAISS: A FAISS VectorStore containing the embedded chunks.
    """

    # Validate inputs
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    if not openai_api_key:
        raise ValueError("OpenAI API key is required")
        
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")

    # Load the PDF
    loader = PyPDFLoader(pdf_path)
    pages = [page for page in loader.lazy_load()]

    # Split the text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                              chunk_overlap=chunk_overlap,
                                              separators=["\n\n", "\n", " ", ""])
    
    docs = []
    all_chunks = []  # To keep track of all chunks across all pages
    for page_num, page in enumerate(pages):
        chunks = splitter.split_text(page.page_content)
        all_chunks.extend(chunks)  # Add all chunks from this page

        # print(f"Page {page_num}: Created {len(chunks)} chunks")

        # Convert each chunk into a "Document" object for LangChain
        for i, chunk in enumerate(chunks):
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "page_number": page.metadata.get("page"),
                    "chunk_index": i,
                    "source": pdf_path
                }
            ))

    '''
        # Print some samples to verify diversity
        print(f"Total chunks created: {len(all_chunks)}")
        print("\nSample chunks:")
        for i in range(min(3, len(all_chunks))):
            print(f"\nChunk {i}:")
            print(all_chunks[i][:150] + "...")  # First 150 chars

        # Print from middle of document
        if len(all_chunks) > 10:
            middle_idx = len(all_chunks) // 2
            print(f"\nMiddle chunk {middle_idx}:")
            print(all_chunks[middle_idx][:150] + "...")

        # Print from end of document
        if len(all_chunks) > 5:
            print(f"\nLast chunk {len(all_chunks)-1}:")
            print(all_chunks[-1][:150] + "...")
    '''

    # Embed the chunks
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)

    # Build the vectorstore
    vectorstore = LangchainPinecone.from_documents(
        docs, 
        embeddings, 
        index_name=index_name
    )

    return vectorstore


def build_retrieval_qa_chain(vectorstore, openai_api_key):
    """Build a retrieval-based QA chain.

    Args:
        vectorstore (Pinecone): The pinecone vectorstore.
        openai_api_key (str): OpenAI API key.

    Returns:
        RetrievalQA: A retrieval-based QA chain.
    """

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) 

    # ChatOpenAI for a better chat-style LLM
    llm = ChatOpenAI(
        openai_api_key=openai_api_key, 
        temperature=0,       # Keep it deterministic for testing
        model_name="gpt-3.5-turbo"  # or "gpt-4"
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True  # If you want the actual sources returned
    )

    return chain


if __name__ == "__main__":
    
    pdf_file = os.environ.get("PDF_FILE")
    openai_key = os.environ.get("OPENAI_API_KEY")
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index_name = os.environ.get("PINECONE_INDEX_NAME")

    if index_name not in pc.list_indexes().names():
        pc.create_index(index_name, dimension=1536,
                        metric='euclidean',
                        spec=ServerlessSpec(
                            cloud='aws',
                            region="us-east-1"
                        ))
    index = pc.Index(index_name)

    try:
        # Create the VectorStore from the PDF
        vectorstore = chunk_and_embed_pdf(pdf_file, openai_key, index_name)
        logger.info(f"Successfully created vectorstore with")
    except Exception as e:
        logger.error(f"Error creating vectorstore: {e}")

    # Quick test: Query the VectorStore
    # query = "What is the main focus of the paper?"
    # docs = vectorstore.similarity_search(query, k=3)

    # print("Top 3 Relevant Chunks:")
    # for i, doc in enumerate(docs, start=1):
    #     print(f"\n--- Chunk #{i} ---")
    #     print("Content:", doc.page_content)
    #     print("Metadata:", doc.metadata)


    # Build chain
    qa_chain = build_retrieval_qa_chain(vectorstore, openai_key)

    # Ask a question about the PDF
    query = "What is the main focus of the paper?"
    result = qa_chain({"query": query})

    # The chain returns a dict with at least:
    #  'result': the LLM's final answer
    #  'source_documents': (only if return_source_documents=True)
    print("Answer:", result["result"])

    print("\nSources:")
    for i, doc in enumerate(result["source_documents"], start=1):
        print(f"--- Source {i} ---")
        print("Page Content (truncated):", doc.page_content[:200], "...")
        print("Metadata:", doc.metadata)